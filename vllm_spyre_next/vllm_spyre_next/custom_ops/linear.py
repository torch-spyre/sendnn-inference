# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific linear layer implementations using out-of-tree (OOT) registration.

This module provides Spyre-device-specific replacements for the parallel linear
layer classes used inside MLP blocks:

    - SpyreMergedColumnParallelLinear  — replaces MergedColumnParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreQKVParallelLinear          — replaces QKVParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreRowParallelLinear          — replaces RowParallelLinear
      (vllm/model_executor/layers/linear.py)

Since tensor_parallel=1 is assumed, both classes are functionally equivalent
to F.linear(input, weight, bias) and share the same implementation pattern.

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.
    - Tensor parallelism: TP=1 assumed (single Spyre device)

References:
    - Upstream linear layers: vllm/model_executor/layers/linear.py
    - Pattern reference:      vllm_spyre_next/custom_ops/rms_norm.py
"""

import torch
import torch.nn.functional as F
from functools import lru_cache

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


class SpyreLinearBase:
    """Shared implementation for Spyre linear layers at TP=1."""

    def _init_spyre_linear(self, layer_prefix: str):
        """Common initialization for Spyre linear layers."""
        if self.tp_size > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports TP=1, got TP={self.tp_size}"
            )

        logger.debug("Building custom %s", self.__class__.__name__)

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        # NOTE: Using torch.compile directly here since PluggableLayer (unlike CustomOp)
        # does not provide a maybe_compile method. This should be revisited in the future
        # to align with vLLM's compilation infrastructure once PluggableLayer supports
        # compilation hooks similar to CustomOp.maybe_compile.
        self.maybe_compiled_forward_spyre = torch.compile(self.forward_spyre, dynamic=False)
        self._layer_name = register_layer(self, layer_prefix)

        logger.warning_once(
            "%s: no dtype promotion (torch-spyre limitation),"
            "expect numerical differences to upstream vLLM.",
            self.__class__.__name__,
        )

    def forward_spyre(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, weight, bias)

    def _forward_spyre_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_device = x.device

        # Bias is fused into F.linear only when not skipping bias add
        bias = self.bias.data if (self.bias is not None and not self.skip_bias_add) else None

        out = self.maybe_compiled_forward_spyre(
            convert(x, self._target_device, self._target_dtype),
            convert(self.weight.data, self._target_device, self._target_dtype),
            convert(bias, self._target_device, self._target_dtype) if bias is not None else None,
        )

        return convert(out, dtype=x_dtype, device=x_device)


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(SpyreLinearBase, MergedColumnParallelLinear):
    """Spyre MergedColumnParallelLinear (TP=1 only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_spyre_linear("spyre_merged_col_linear")

    # `MergedColumnParallelLinear` is a PluggableLayer and we register a class as OOT,
    # thus, the `forward` method is invoked when the OOT is triggered.
    def forward(self, input_: torch.Tensor):
        if input_.device.type == "spyre":
            output = self._forward_spyre_impl(input_)
        else:
            output = input_.new_empty(
                input_.shape[0],
                self.output_size_per_partition,
            )
            torch.ops.vllm.spyre_merged_col_linear(input_, output, self._layer_name)

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreLinearBase, QKVParallelLinear):
    """Spyre QKVParallelLinear (TP=1 only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_spyre_linear("spyre_qkv_parallel_linear")

    def forward(self, input_: torch.Tensor):
        if input_.device.type == "spyre":
            output = self._forward_spyre_impl(input_)
            # D2H before downstream .split() — Spyre can't handle strided views
            output = convert(output, device="cpu")
        else:
            output = input_.new_empty(
                input_.shape[0],
                self.output_size_per_partition,
            )
            torch.ops.vllm.spyre_qkv_parallel_linear(input_, output, self._layer_name)

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreLinearBase, RowParallelLinear):
    """Spyre RowParallelLinear (TP=1 only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_spyre_linear("spyre_row_parallel_linear")

    # `SpyreRowParallelLinear` is a PluggableLayer and we register a class as OOT,
    # thus, the `forward` method is invoked when the OOT is triggered.
    def forward(self, input_: torch.Tensor):
        if input_.device.type == "spyre":
            output = self._forward_spyre_impl(input_)
        else:
            output = input_.new_empty(
                *input_.shape[:-1],
                self.output_size_per_partition,
            )
            torch.ops.vllm.spyre_row_parallel_linear(input_, output, self._layer_name)
            # Always output on Spyre — needed for residual add with Spyre hidden_states
            output = convert(output, device=self._target_device, dtype=self._target_dtype)

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def _make_spyre_linear_op_func(op_name: str):
    def _op_func(
        x: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        layer = get_layer(layer_name)
        result = layer._forward_spyre_impl(x)
        output.copy_(result)

    _op_func.__name__ = f"_{op_name}_op_func"
    return _op_func


@lru_cache(maxsize=1)
def register():
    """Register Spyre linear custom ops."""
    for op_name in [
        "spyre_merged_col_linear",
        "spyre_qkv_parallel_linear",
        "spyre_row_parallel_linear",
    ]:
        direct_register_custom_op(
            op_name=op_name,
            op_func=_make_spyre_linear_op_func(op_name),
            mutates_args=["output"],
            fake_impl=_fake_impl,
        )
        logger.info("Registered custom op: %s", op_name)
