# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific RMSNorm implementation using out-of-tree (OOT) registration.

This module provides a custom RMSNorm layer for IBM's Spyre device,
replacing the upstream vLLM implementation (vllm/model_executor/layers/layernorm.py)
when instantiated.

Architecture:
    - OOT Registration: @RMSNorm.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch, calls custom op for
      torch.compile opacity
    - Custom Op Boundary: torch.ops.vllm.spyre_rmsnorm is opaque to torch.compile,
      so _forward_spyre_impl runs eagerly outside the compiled graph
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
import torch.utils._pytree as pytree

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.layernorm import RMSNorm
from functools import lru_cache

from .utils import convert, register_layer, get_layer, _fake_impl

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
        """OOT forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_rmsnorm which retrieves this layer
        from the layer registry and calls _forward_spyre_impl outside
        the compilation graph. This prevents torch.compile from inlining the
        Spyre-specific operations.

        Args:
            x: Input tensor [batch_size, hidden_size]
            residual: Optional residual tensor

        Returns:
            Normalized output, or (output, residual) tuple if residual provided
        """
        output = torch.empty_like(x)
        residual_out = torch.empty_like(residual) if residual is not None else None

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_rmsnorm(x, output, self._layer_name, residual, residual_out)

        if residual is not None:
            return output, residual_out
        return output

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
        """Spyre device execution with padding, device transfer, and dtype conversion.

        Handles Spyre-specific constraints:
            1. Minimum batch size: Pads to 64 if needed
            2. Device transfer: CPU -> Spyre convert to float16
            3. Kernel execution: Calls compiled maybe_compiled_forward_spyre
            4. Result transfer: Spyre -> CPU, trim padding, convert to input dtype

        Limitations:
            - variance_size_override not implemented (raises NotImplementedError)

        Args:
            x: Input tensor [batch_size, hidden_size] on CPU
            residual: Optional residual

        Returns:
            Normalized output [batch_size, hidden_size] in input dtype
        """
        x_dtype = x.dtype
        x_device = x.device

        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        orig_batch_size = x.shape[0]

        # Pad to minimum batch size of 64 (Spyre constraint)
        # Pad at END so original data stays at indices [0:orig_batch_size]
        if x.shape[0] < _SPYRE_MIN_BATCH_SIZE:
            pad_amount = _SPYRE_MIN_BATCH_SIZE - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))
            if residual is not None:
                residual = torch.nn.functional.pad(residual, (0, 0, 0, pad_amount))

        # Execute compiled kernel on Spyre device
        outs = self.maybe_compiled_forward_spyre(
            convert(x, self._target_device, self._target_dtype),
            self.variance_epsilon,
            self.hidden_size,
            convert(self.weight.data, self._target_device, self._target_dtype)
            if self.has_weight
            else None,
            convert(residual, self._target_device, self._target_dtype),
        )

        # Transfer back to CPU and restore original shape
        return pytree.tree_map(
            lambda el: convert(el, dtype=x_dtype, device=x_device)[:orig_batch_size, :],
            outs,
        )


def _op_func(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    residual: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> None:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    result = layer._forward_spyre_impl(x, residual)

    if residual is not None:
        output_data, residual_data = result
        output.copy_(output_data)
        residual_out.copy_(residual_data)
    else:
        output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rmsnorm custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_rmsnorm",
        op_func=_op_func,
        mutates_args=["output", "residual_out"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreRMSNorm")
