# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific VocabParallelEmbedding implementation using out-of-tree (OOT) registration.

This module provides a custom VocabParallelEmbedding layer for IBM's Spyre device,
replacing the upstream vLLM implementation
(vllm/model_executor/layers/vocab_parallel_embedding.py) when instantiated.

Architecture:
    - OOT Registration: @VocabParallelEmbedding.register_oot() replaces upstream
      at instantiation
    - forward_oot(): Entry point for OOT dispatch, calls custom op for
      torch.compile opacity
    - Custom Op Boundary: torch.ops.vllm.spyre_vocab_parallel_embedding is opaque
      to torch.compile, so _forward_spyre_impl runs eagerly outside the compiled graph
    - Separate Compilation: forward_spyre is compiled independently via maybe_compile

Spyre Device Constraints:
    - Algorithm: From vLLM's perspective, embedding runs on Spyre. Internally,
      torch-spyre currently falls back to CPU for aten.embedding.default
      (spyre__embedding) since Spyre does not yet support indirect indexing.
      See: https://github.com/torch-spyre/torch-spyre/issues/420

Limitations:
    - No Tensor Parallelism (TP) support: tp_size > 1 raises NotImplementedError.
    - No quantization support: quant_config != None raises NotImplementedError.

References:
    - Upstream VocabParallelEmbedding:
      vllm/model_executor/layers/vocab_parallel_embedding.py
"""

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from functools import lru_cache

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device.

    This replaces the upstream vLLM VocabParallelEmbedding when instantiated,
    providing Spyre-specific device handling.

    No Tensor Parallelism (TP) is supported: tp_size > 1 raises NotImplementedError.
    No quantization is supported: quant_config != None raises NotImplementedError.
    """

    _dynamic_arg_dims = {"x": [], "weight": []}

    def __init__(self, *args, **kwargs):
        """Initialize SpyreVocabParallelEmbedding layer.

        Compiles the Spyre kernel and registers this instance for custom op lookup.

        Raises:
            NotImplementedError: If tp_size > 1 or quant_config is not None,
                as these are not supported in the current Spyre implementation.
        """
        super().__init__(*args, **kwargs)

        # Check for unsupported configurations after super().__init__
        # sets up the parallel config.
        quant_config = kwargs.get("quant_config")
        if quant_config is not None:
            raise NotImplementedError(
                "SpyreVocabParallelEmbedding does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        if self.tp_size > 1:
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support Tensor Parallelism "
                f"(tp_size={self.tp_size}). TP masking and all_reduce are not implemented. "
                "Only tp_size=1 is supported."
            )

        logger.debug("Building custom VocabParallelEmbedding for Spyre")

        self._target_device = torch.device("spyre")
        # The inputs for spyre need to be torch.int64
        self._input_target_dtype = torch.int64
        # The weights for spyre need to be torch.float16
        self._weight_target_dtype = torch.float16
        self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)

        self._layer_name = register_layer(self, "spyre_vocab_parallel_embedding")

        logger.debug_once(
            "SpyreVocabParallelEmbedding: Dispatch: enabled=%s, Forward method=%s, Compiled=%s",
            self.enabled(),
            self._forward_method.__name__,
            self.maybe_compiled_forward_spyre is not self.forward_spyre,
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """OOT forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_vocab_parallel_embedding which
        retrieves this layer from the layer registry and calls
        _forward_spyre_impl outside the compilation graph. This prevents
        torch.compile from inlining the Spyre-specific operations.

        Args:
            x: Token index tensor [num_tokens] (int64)

        Returns:
            Embedding output [num_tokens, embedding_dim] in weight dtype
        """
        output = torch.empty(
            *x.shape,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=x.device,
        )

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_vocab_parallel_embedding(x, output, self._layer_name)

        return output

    @staticmethod
    def forward_spyre(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Spyre embedding kernel compiled via maybe_compile.

        Args:
            x: Token index tensor [num_tokens] (int64)
            weight: Embedding weight tensor [vocab_size, embedding_dim]

        Returns:
            Embedding output [num_tokens, embedding_dim]
        """
        return F.embedding(x, weight)

    def _forward_spyre_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Spyre device execution: device transfer, kernel call, result transfer.

        From vLLM's perspective, this operation runs on the Spyre device.
        Internally, torch-spyre currently falls back to CPU for embedding
        (aten.embedding.default -> spyre__embedding) since Spyre does not
        yet support indirect indexing (gather/scatter).

        Note: Once torch-spyre gains native embedding support, this fallback
        will be removed. See: https://github.com/torch-spyre/torch-spyre/issues/420

        No TP masking or all_reduce is performed (tp_size > 1 is not supported).

        Args:
            x: Token index tensor [num_tokens] (int64)

        Returns:
            Embedding output [num_tokens, embedding_dim] in weight dtype
        """
        out_dtype = self.weight.dtype
        out_device = x.device

        out = self.maybe_compiled_forward_spyre(
            convert(x, dtype=self._input_target_dtype, device=self._target_device),
            convert(self.weight.data, dtype=self._weight_target_dtype, device=self._target_device),
        )

        # Transfer back to original device and restore original dtype
        return pytree.tree_map(
            lambda el: convert(el, dtype=out_dtype, device=out_device),
            out,
        )


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
    """Register the spyre_vocab_parallel_embedding custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_vocab_parallel_embedding",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreVocabParallelEmbedding")
