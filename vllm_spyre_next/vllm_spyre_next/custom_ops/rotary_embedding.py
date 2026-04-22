# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacement for RotaryEmbedding (CPU fallback).

Remove this file once Spyre natively supports rotary embedding ops.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from functools import lru_cache

from .utils import convert

logger = init_logger(__name__)


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """OOT RotaryEmbedding that falls back to CPU execution.

    Keeps cos_sin_cache on CPU via an _apply no-op. Inputs are moved to
    CPU for computation, and outputs are copied back to the original device.
    """

    def _apply(self, fn, recurse=True):
        # Keep cos_sin_cache on CPU so forward_native can use it directly.
        return self

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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


@lru_cache(maxsize=1)
def register():
    # OOT class registration happens at import time via the decorator.
    pass
