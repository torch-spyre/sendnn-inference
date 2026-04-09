# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific RMSNorm implementation using out-of-tree (OOT) registration.

This module provides a custom RMSNorm layer for IBM's Spyre device,
replacing the upstream vLLM implementation (vllm/model_executor/layers/layernorm.py)
when instantiated.

Architecture:
    - OOT Registration: @RMSNorm.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch, calls _forward_spyre_impl
      directly (no custom op boundary needed since Spyre does not support
      in-device tensor copy)
    - Separate Compilation: forward_spyre is compiled independently via maybe_compile

Spyre Device Constraints:
    - Minimum batch size: 64 (due to spyre constraint, automatically padded)
    - Device dtype: float16 (converted for CPU)
    - Output dtype: bfloat16 (converted on CPU)
    - Algorithm: Transpose-based computation with torch.ops.spyre.full()

Limitations:
    Currently the implementation in `forward_spyre` is similar to the
    upstream implementation in `forward_static` from vllm/model_executor/layers/layernorm.py,
    but it DOES NOT use the promotion of the data types, as this is not
    yet supported in torch-spyre.

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

import torch
import torch.utils._pytree as pytree

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from .utils import register_layer, convert

logger = init_logger(__name__)

# Minimum batch size required by Spyre hardware.
_SPYRE_MIN_BATCH_SIZE = 64


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre device.

    This replaces the upstream vLLM RMSNorm (vllm/model_executor/layers/layernorm.py)
    when instantiated, providing Spyre-specific optimizations and device handling.
    """

    _dynamic_arg_dims = {"x": [], "residual": []}

    def __init__(self, *args, **kwargs):
        """Initialize SpyreRMSNorm layer.

        Compiles the Spyre kernel based on VLLM_SPYRE_NEXT_RMSNORM_KERNEL
        environment variable and registers this instance in static_forward_context.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom RMS norm")

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16
        self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)

        self._layer_name = register_layer(self, "spyre_rmsnorm")

        logger.warning_once(
            "SpyreRMSNorm: no dtype promotion is performed, "
            "expect numerical differences to upstream vLLM."
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """OOT forward pass — calls _forward_spyre_impl directly.

        No custom op boundary is used because the Spyre runtime does not
        support in-device tensor copy_ or returning Spyre tensors from
        custom ops (triggers D2H copy in the dispatch machinery).

        Args:
            x: Input tensor [batch_size, hidden_size]
            residual: Optional residual tensor

        Returns:
            Normalized output, or (output, residual) tuple if residual provided
        """
        return self._forward_spyre_impl(x, residual)

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RMS norm using transpose-based computation (active implementation).

        Based on upstream vLLM's forward_static (vllm/model_executor/layers/layernorm.py)
        but adapted for Spyre device with transpose operations and torch.ops.spyre.full().
        Compiled separately via torch.compile in __init__.

        Key differences from upstream:
            - Uses transpose(-1, -2) for computation efficiency on Spyre
            - Creates epsilon tensor via torch.ops.spyre.full() instead of scalar
            - No dtype promotion support (torch-spyre limitation)
        """
        if residual is not None:
            x = x + residual
            residual = x

        if x.shape[-1] != hidden_size:
            raise ValueError(f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}")

        variance_epsilon = torch.full(
            x.shape, variance_epsilon, dtype=torch.float16, device=x.device
        )

        variance = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + variance_epsilon)

        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        else:
            return x, residual

    def _forward_spyre_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Spyre device execution with padding and kernel dispatch.

        Handles Spyre-specific constraints:
            1. Minimum batch size: Pads to 64 if needed
            2. Kernel execution: Calls compiled maybe_compiled_forward_spyre

        Limitations:
            - variance_size_override not implemented (raises NotImplementedError)

        Args:
            x: Input tensor [batch_size, hidden_size]
            residual: Optional residual tensor

        Returns:
            Normalized output, or (output, residual) tuple if residual provided.
        """
        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        # Execute compiled kernel on Spyre device
        return self.maybe_compiled_forward_spyre(
            x,
            self.variance_epsilon,
            self.hidden_size,
            self.weight.data if self.has_weight else None,
            residual,
        )
        


def register():
    """No-op: custom op registration is not needed when forward_oot calls
    _forward_spyre_impl directly (Spyre does not support in-device copy_
    or returning tensors from custom ops)."""
    pass
