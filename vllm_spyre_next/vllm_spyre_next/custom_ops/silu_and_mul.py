# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific SiluAndMul implementation using out-of-tree (OOT) registration.

This module provides a custom SiluAndMul (SwiGLU) activation layer for
IBM's Spyre device, replacing the upstream vLLM implementation from
vllm/model_executor/layers/activation.py when instantiated.

Architecture:
    - OOT Registration: @SiluAndMul.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch, calls custom op for
      torch.compile opacity
    - Custom Op Boundary: torch.ops.vllm.spyre_siluandmul is opaque to torch.compile,
      so _forward_spyre_impl runs eagerly outside the compiled graph
    - Separate Compilation: forward_spyre is compiled independently via maybe_compile

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.

Output Shape Note:
    Unlike RMSNorm (same input/output shape), SiluAndMul halves the last dimension:
    input shape: [..., 2*d] -> output shape: [..., d]

References:
    - Upstream SiluAndMul: vllm/model_executor/layers/activation.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.activation import SiluAndMul
from functools import lru_cache

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device.

    This replaces the upstream vLLM SiluAndMul (vllm/model_executor/layers/activation.py)
    when instantiated, providing Spyre-specific optimizations and device handling.

    Computes: x -> silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2
    """

    _dynamic_arg_dims = {"x1": [], "x2": []}

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer.

        Compiles the Spyre kernel and registers this instance in static_forward_context.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom SiluAndMul")

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16
        self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)

        self._layer_name = register_layer(self, "spyre_siluandmul")

        logger.debug_once(
            "SpyreSiluAndMul: Dispatch: enabled=%s, Forward method=%s, Compiled=%s",
            self.enabled(),
            self._forward_method.__name__,
            self.maybe_compiled_forward_spyre is not self.forward_spyre,
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """OOT forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_siluandmul which retrieves this layer
        from the layer registry and calls _forward_spyre_impl outside
        the compilation graph.

        Args:
            x: Input tensor [..., 2*d]

        Returns:
            Activated output tensor [..., d]
        """
        d = x.shape[-1] // 2
        output = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_siluandmul(x, output, self._layer_name)

        return output

    @staticmethod
    def forward_spyre(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Spyre-optimized silu+multiply kernel compiled via torch.compile.

        Computes silu(x1) * x2 on the Spyre device, relying on torch-spyre's
        registered aten::silu.out kernel. The two halves are passed in as
        separate tensors because the Spyre device does not yet support tensor
        slicing (strided views); the split is therefore performed on CPU before
        this method is called (see _forward_spyre_impl).

        Args:
            x1: First half of the gated input, shape [..., d], on Spyre device
                (float16).  silu is applied to this half.
            x2: Second half of the gated input, shape [..., d], on Spyre device
                (float16).  Acts as the multiplicative gate.

        Returns:
            Output tensor of shape [..., d] on the Spyre device (float16).
        """
        return F.silu(x1) * x2

    def _forward_spyre_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Spyre device execution: CPU slicing workaround, device transfer, kernel call.

        The Spyre device does not currently support strided tensor views (slicing),
        so the input is split into its two halves on the CPU before being
        transferred to the device.  Once tensor slicing is supported this method
        should revert to the simpler single-tensor path (see commented-out block).

        Execution steps:
            1. Slice on CPU: split x into x1 = x[..., :d] and x2 = x[..., d:]
            2. Device transfer: convert x1 and x2 independently to Spyre (float16)
               via convert_for_spyre
            3. Kernel execution: call compiled maybe_compiled_forward_spyre(x1_spyre, x2_spyre)
            4. Result transfer: Spyre -> original device, restore original dtype

        Args:
            x: Input tensor of shape [..., 2*d] on CPU with arbitrary float dtype.

        Returns:
            Activated output tensor of shape [..., d] on the original device with
            the original dtype.
        """
        x_dtype = x.dtype
        x_device = x.device

        # Note: Workaround with tensor slicing on CPU
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        out = self.maybe_compiled_forward_spyre(
            convert(x1, self._target_device, self._target_dtype),
            convert(x2, self._target_device, self._target_dtype),
        )

        # Transfer back to original device and restore original dtype
        return convert(out, x_device, x_dtype)


def _op_func(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    result = layer._forward_spyre_impl(x)
    output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_siluandmul custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_siluandmul",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreSiluAndMul")
