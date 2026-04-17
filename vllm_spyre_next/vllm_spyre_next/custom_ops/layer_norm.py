# SPDX-License-Identifier: Apache-2.0

import torch
from functools import lru_cache

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.layernorm import LayerNorm

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)

_SPYRE_MIN_BATCH_SIZE = 64


@LayerNorm.register_oot(name="LayerNorm")
class SpyreLayerNorm(LayerNorm):
    """
    Spyre implementation of LayerNorm.

    This mirrors the structure used in SpyreRMSNorm but implements
    standard LayerNorm normalization using mean and variance.
    """

    _dynamic_arg_dims = {"x": []}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        self.maybe_compiled_forward_spyre = self.maybe_compile(
            self.forward_spyre
        )

        self._layer_name = register_layer(self, "spyre_layernorm")

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        eps: float,
        hidden_size: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        """
        Core LayerNorm math implementation.
        """

        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_norm = (x - mean) * torch.rsqrt(variance + eps)

        if weight is not None:
            x_norm = x_norm * weight

        if bias is not None:
            x_norm = x_norm + bias

        return x_norm

    def _forward_spyre_impl(self, x: torch.Tensor):

        x_dtype = x.dtype
        x_device = x.device

        orig_batch_size = x.shape[0]

        # Pad if batch smaller than spyre minimum
        if x.shape[0] < _SPYRE_MIN_BATCH_SIZE:
            pad_amount = _SPYRE_MIN_BATCH_SIZE - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))

        out = self.maybe_compiled_forward_spyre(
            convert(x, self._target_device, self._target_dtype),
            self.eps,
            self.dim,
            convert(self.weight.data, self._target_device, self._target_dtype),
            convert(self.bias.data, self._target_device, self._target_dtype)
            if self.bias is not None
            else None,
        )

        return convert(out, dtype=x_dtype, device=x_device)[:orig_batch_size]


def _op_func(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
):
    """
    Custom op entry point used by Spyre backend.
    """

    layer = get_layer(layer_name)

    result = layer._forward_spyre_impl(x)

    output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """
    Register the custom operation with torch.
    """

    direct_register_custom_op(
        op_name="spyre_layernorm",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )

    logger.info("Registered custom op: SpyreLayerNorm")
