# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre CPU fallback for RotaryEmbedding.

Remove this file once Spyre supports rotary embedding natively.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .cpu_fallback import SpyreCpuFallbackMixin
from .utils import _fake_impl, convert, register_spyre_dispatch

logger = init_logger(__name__)


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(SpyreCpuFallbackMixin, RotaryEmbedding):
    """OOT RotaryEmbedding that falls back to CPU execution.

    Keeps cos_sin_cache on CPU (via _apply no-op from SpyreCpuFallbackMixin).
    Inputs arrive on Spyre, are moved to CPU for computation, and outputs
    are copied back to Spyre.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_cpu_fallback("rotary_embedding")

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass — bypasses custom op when on Spyre.

        When inputs are on Spyre, calls forward_impl_direct which executes
        on CPU and returns results converted to the original device, avoiding
        the custom op boundary (Spyre does not support in-device copy_).

        Falls back to custom op path for CPU inputs.
        """
        if query.device.type == "spyre":
            return self.forward_impl_direct(positions, query, key)

        output_query = torch.empty_like(query)

        if key is not None:
            output_key = torch.empty_like(key)
            torch.ops.vllm.spyre_rotary_embedding(
                positions,
                query,
                key,
                output_query,
                output_key,
                self.prefix,
            )
            return output_query, output_key

        # key=None case (e.g. cross-layer KV sharing)
        torch.ops.vllm.spyre_rotary_embedding_q_only(
            positions,
            query,
            output_query,
            self.prefix,
        )
        return output_query, None

    def forward_impl_direct(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Direct forward: CPU execution, returns results on original device.

        Bypasses custom op boundary entirely. Moves inputs to CPU, calls
        upstream forward_native, converts results back to the input device.
        """
        target_device = query.device
        target_dtype = query.dtype

        cpu_positions = convert(positions, device="cpu")
        cpu_query = convert(query, device="cpu")
        cpu_key = convert(key, device="cpu")

        result_query, result_key = RotaryEmbedding.forward_native(
            self,
            cpu_positions,
            cpu_query,
            cpu_key,
        )

        out_query = convert(result_query, device=target_device, dtype=target_dtype)
        out_key = (
            convert(result_key, device=target_device, dtype=target_dtype)
            if result_key is not None
            else None
        )
        return out_query, out_key

    def forward_impl(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        output_query: torch.Tensor,
        output_key: torch.Tensor | None,
    ) -> None:
        """Implementation called by custom op, executes outside torch.compile.

        Moves inputs to CPU, calls upstream forward_native, copies results
        back to pre-allocated output tensors (on Spyre).
        """
        cpu_positions = convert(positions, device="cpu")
        cpu_query = convert(query, device="cpu")
        cpu_key = convert(key, device="cpu")

        result_query, result_key = RotaryEmbedding.forward_native(
            self,
            cpu_positions,
            cpu_query,
            cpu_key,
        )

        output_query.copy_(result_query)
        if output_key is not None and result_key is not None:
            output_key.copy_(result_key)


# --- Custom op: with key ---


def spyre_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    output_query: torch.Tensor,
    output_key: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op — rotary embedding with query and key."""
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(positions, query, key, output_query, output_key)


# --- Custom op: query only (key=None) ---


def spyre_rotary_embedding_q_only(
    positions: torch.Tensor,
    query: torch.Tensor,
    output_query: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op — rotary embedding with query only (no key)."""
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(positions, query, None, output_query, None)


def register():
    """Register rotary embedding custom ops."""
    direct_register_custom_op(
        op_name="spyre_rotary_embedding",
        op_func=spyre_rotary_embedding,
        mutates_args=["output_query", "output_key"],
        fake_impl=_fake_impl,
    )
    register_spyre_dispatch("spyre_rotary_embedding", spyre_rotary_embedding)
    direct_register_custom_op(
        op_name="spyre_rotary_embedding_q_only",
        op_func=spyre_rotary_embedding_q_only,
        mutates_args=["output_query"],
        fake_impl=_fake_impl,
    )
    register_spyre_dispatch("spyre_rotary_embedding_q_only", spyre_rotary_embedding_q_only)
    logger.info("Registered custom op: spyre_rotary_embedding")
