# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-card unit test for sink attention on the CP path of DotProductAttention.

The CP branch in ``DotProductAttention`` (when ``context_parallel_size > 1``)
expands ``attn_mask_startend_row_indices`` via
``expand_attn_mask_startend_row_indices_for_cp`` and dispatches to
``flashmask_attention_cp``. Internally, on FA4 builds, that helper all-gathers
K/V across the CP group and calls ``_flash_attn_fwd(..., learnable_sink=sink)``.

Equivalently, the cp=1 branch of the same layer calls
``flashmask_attention(..., sink=softmax_offset)`` from ``flash_mask_facade``
directly. Numerically the two routes coincide on FA4 modulo the
gather/scatter, so we use the latter as the full-sequence baseline.

This test scatters Q/K/V across the CP group (balanced dual-chunk layout via
``scatter_balance``), runs ``DotProductAttention`` on each rank, and checks
that the all-gathered output and gradients match the single-rank full-sequence
reference within bf16 tolerance. Sink grad is compared after an all-reduce
across the CP group (each rank only owns a partial contribution to it).

Run with:
    FLAGS_flash_attn_version=4 \\
    python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \\
        tests/multi_card_tests/test_cp_dot_product_attention_sink.py
"""

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddlefleet_ops.flash_mask_facade import flashmask_attention

from paddlefleet.context_parallel_utils import (
    all_gather_balance,
    scatter_balance,
)
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


def _make_config(context_parallel_size=1):
    return TransformerConfig(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        softmax_scale=None,
        use_bias=True,
        recompute_granularity=None,
        recompute_modules=None,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, 2.0),
        rms_norm_eps=1e-5,
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
        apply_query_key_layer_scaling=False,
        sliding_window=None,
        window_attn_skip_freq=None,
        fp16=False,
        bf16=True,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
        attention_dropout=0.0,
        softmax_type="learnable",
        fa_version=4,
        params_dtype=paddle.bfloat16,
    )


def _initialize_cp_fleet(cp_size):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": cp_size,
        "sep_degree": 1,
        "cp_degree": cp_size,
        "ep_degree": cp_size,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy=strategy)


def _make_full_qkv_and_sink():
    # Same seed on every rank -> same full tensors before the scatter.
    paddle.seed(2026)
    np.random.seed(2026)
    batch_size = 1
    seq_len = 4096
    num_heads = 4
    head_dim = 64
    query = paddle.randn(
        [batch_size, seq_len, num_heads, head_dim], dtype=paddle.bfloat16
    )
    key = paddle.randn(
        [batch_size, seq_len, num_heads, head_dim], dtype=paddle.bfloat16
    )
    value = paddle.randn(
        [batch_size, seq_len, num_heads, head_dim], dtype=paddle.bfloat16
    )
    sink = paddle.randn([num_heads], dtype=paddle.bfloat16)
    return query, key, value, sink


def _build_no_mask_startend_row_indices(seq_len):
    """Reproduce ``expand_attn_mask_startend_row_indices_for_cp(None, ...)``.

    With ``attn_mask_startend_row_indices=None``, the CP helper builds
    ``[b=1, h=1, S, 1]`` filled with ``S`` then concats ``arange(S)`` along
    the last axis, producing ``[1, 1, S, 2]``. The CP path and the
    full-sequence baseline must use the *exact same* indices for the
    comparison to be meaningful.
    """
    col0 = paddle.full(
        shape=[1, 1, seq_len, 1], fill_value=seq_len, dtype=paddle.int32
    )
    col1 = paddle.arange(seq_len, dtype=paddle.int32).reshape(
        [1, 1, seq_len, 1]
    )
    return paddle.concat([col0, col1], axis=-1).cuda()


def _set_sink(attn, sink):
    with paddle.no_grad():
        attn.softmax_offset.set_value(sink.astype(attn.softmax_offset.dtype))
    attn.softmax_offset.stop_gradient = False


def _clone_for_grad(tensor):
    cloned = tensor.detach().clone()
    cloned.stop_gradient = False
    return cloned


class TestCPDotProductAttentionSink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("Requires CUDA for CP flashmask attention.")
        cls.world_size = dist.get_world_size()
        if cls.world_size < 2:
            raise unittest.SkipTest("Requires at least 2 ranks.")
        _initialize_cp_fleet(cls.world_size)

    def test_forward_backward_matches_full_sequence_sink_attention(self):
        rank = dist.get_rank()
        paddle.device.set_device(f"gpu:{rank}")

        query_seed, key_seed, value_seed, sink_seed = _make_full_qkv_and_sink()
        query_seed = query_seed.cuda()
        key_seed = key_seed.cuda()
        value_seed = value_seed.cuda()
        sink_seed = sink_seed.cuda()

        seq_len_full = query_seed.shape[1]
        no_mask_indices = _build_no_mask_startend_row_indices(seq_len_full)

        # --------------------------------------------------------------
        # Full-sequence baseline. This is exactly the call DotProductAttention
        # makes when context_parallel_size == 1 with a learnable sink:
        #   flashmask_attention(q, k, v, startend_row_indices=...,
        #                       dropout=0.0, causal=False, sink=sink)
        # On FA4 it routes to the same kernel that
        # cp_flashmask_allgatherkv_balance_forward drives internally
        # after the all-gather of K/V.
        # --------------------------------------------------------------
        query_full = _clone_for_grad(query_seed)
        key_full = _clone_for_grad(key_seed)
        value_full = _clone_for_grad(value_seed)
        baseline_sink = sink_seed.detach().clone()
        baseline_sink.stop_gradient = False

        expected = flashmask_attention(
            query_full,
            key_full,
            value_full,
            startend_row_indices=no_mask_indices,
            dropout=0.0,
            causal=False,
            sink=baseline_sink,
        )
        # flashmask_attention returns [B, S, H, D]; flatten the last two dims
        # to match DotProductAttention's flashmask-path output shape.
        expected = expected.reshape([expected.shape[0], expected.shape[1], -1])
        expected.astype("float32").sum().backward()

        # --------------------------------------------------------------
        # CP path. Each rank receives a balanced dual-chunk slice of Q/K/V
        # and lets DotProductAttention's CP branch handle the all-gather
        # internally via flashmask_attention_cp.
        # --------------------------------------------------------------
        cp_group = (
            fleet.get_hybrid_communicate_group().get_context_parallel_group()
        )
        query = _clone_for_grad(
            scatter_balance(query_seed, axis=1, group=cp_group)
        )
        key = _clone_for_grad(scatter_balance(key_seed, axis=1, group=cp_group))
        value = _clone_for_grad(
            scatter_balance(value_seed, axis=1, group=cp_group)
        )
        query.retain_grads()
        key.retain_grads()
        value.retain_grads()

        cp_attn = DotProductAttention(
            config=_make_config(context_parallel_size=self.world_size),
            layer_number=1,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="self",
        )
        cp_attn.eval()
        _set_sink(cp_attn, sink_seed)

        actual_local = cp_attn(
            query,
            key,
            value,
            None,  # attention_mask
            attn_mask_type=AttnMaskType.no_mask,
        )
        actual_local.astype("float32").sum().backward()
        actual = all_gather_balance(actual_local, axis=1, group=cp_group)

        np.testing.assert_allclose(
            np.array(actual.astype("float32")),
            np.array(expected.astype("float32")),
            rtol=3e-2,
            atol=3e-2,
        )

        # Input grads: gather local slices and compare to the full-sequence
        # reference grad.
        query_grad = all_gather_balance(query.grad, axis=1, group=cp_group)
        key_grad = all_gather_balance(key.grad, axis=1, group=cp_group)
        value_grad = all_gather_balance(value.grad, axis=1, group=cp_group)
        for actual_grad, expected_grad, name in [
            (query_grad, query_full.grad, "query_grad"),
            (key_grad, key_full.grad, "key_grad"),
            (value_grad, value_full.grad, "value_grad"),
        ]:
            np.testing.assert_allclose(
                np.array(actual_grad.astype("float32")),
                np.array(expected_grad.astype("float32")),
                rtol=3e-2,
                atol=3e-2,
                err_msg=name,
            )

        # Sink grad: each CP rank only sees the contribution from its own
        # query slice, so we all-reduce across the CP group before comparing
        # to the baseline (which used the full sequence on a single rank).
        sink_grad = cp_attn.softmax_offset.grad.clone()
        dist.all_reduce(sink_grad, group=cp_group)
        np.testing.assert_allclose(
            np.array(sink_grad.astype("float32")),
            np.array(baseline_sink.grad.astype("float32")),
            rtol=3e-2,
            atol=3e-2,
            err_msg="sink_grad",
        )


if __name__ == "__main__":
    unittest.main()
