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
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.
    - Epsilon as tensor: Instead of a scalar, a tensor is created via torch.full()

Limitations:
    Currently the implementation in `forward_spyre` is similar to the
    upstream implementation in `forward_static` from vllm/model_executor/layers/layernorm.py,
    but it DOES NOT use the promotion of the data types, as this is not
    yet supported in torch-spyre.

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from .utils import convert, register_layer

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
        logger.debug_once(
            "SpyreRMSNorm: Dispatch: enabled=%s, Forward method=%s, Compiled=%s",
            self.enabled(),
            self._forward_method.__name__,
            self.maybe_compiled_forward_spyre is not self.forward_spyre,
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RMS norm implementation.

        Based on upstream vLLM's forward_static (vllm/model_executor/layers/layernorm.py)
        but adapted for Spyre device. Compiled separately via torch.compile in __init__.

        Key differences from upstream:
            - Creates epsilon tensor via torch.full() instead of scalar
            - No dtype promotion support to torch.float32 (torch-spyre limitation)
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
            Normalized output [batch_size, hidden_size] in input dtype
        """
        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        out_dtype = x.dtype
        out_device = x.device

        # Execute compiled kernel on Spyre device
        outs = self.maybe_compiled_forward_spyre(
            convert(x, self._target_device, self._target_dtype),
            self.variance_epsilon,
            self.hidden_size,
            convert(self.weight.data, self._target_device, self._target_dtype)
            if self.has_weight
            else None,
            convert(residual, self._target_device, self._target_dtype)
            if residual is not None
            else None,
        )

        # Convert back to original device/dtype
        if isinstance(outs, tuple):
            return tuple(convert(o, device=out_device, dtype=out_dtype) for o in outs)
        return convert(outs, device=out_device, dtype=out_dtype)


def register():
    """No-op: custom op registration is not needed when forward_oot calls
    _forward_spyre_impl directly (Spyre does not support in-device copy_
    or returning tensors from custom ops)."""
    pass
