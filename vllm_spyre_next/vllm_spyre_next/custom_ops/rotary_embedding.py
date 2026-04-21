# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific RotaryEmbedding implementation using out-of-tree (OOT) registration.

This module provides a custom RoPE (Rotary Position Embedding) layer for IBM's Spyre device,
replacing the upstream vLLM implementation (vllm/model_executor/layers/rotary_embedding.py)
when instantiated.

Architecture:
    - OOT Registration: @RotaryEmbedding.register_oot() replaces upstream at instantiation
    - Custom Op Boundary: torch.ops.vllm.spyre_rotary_embedding is opaque to torch.compile,
      so _forward_spyre_impl runs eagerly outside the compiled graph
    - Separate Compilation: forward_spyre is compiled independently via maybe_compile

Spyre Device Constraints:
    - Device dtype: float16 (converted for Spyre)
    - Output dtype: matches input dtype (converted on CPU)
    - Cache management: cos/sin caches stored on Spyre device

Limitations:
    - No dtype promotion (torch-spyre limitation)
    - rope_scaling not yet implemented
    - sin/cos calculation use float32 dtype for accuracy

References:
    - Upstream RotaryEmbedding: vllm/model_executor/layers/rotary_embedding.py
"""

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from functools import lru_cache
import torch.utils._pytree as pytree

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """Out-of-tree (OOT) RotaryEmbedding implementation for IBM's Spyre device.

    This replaces the upstream vLLM RotaryEmbedding when instantiated,
    providing Spyre-specific optimizations and device handling.

    Implements RoPE (Rotary Position Embedding) which applies position-dependent
    rotations to query and key tensors in attention mechanisms.
    """

    _dynamic_arg_dims = {"positions": [], "query": [], "key": []}

    def __init__(self, *args, **kwargs):
        """Initialize SpyreRotaryEmbedding layer.
        Compiles the Spyre kernel and registers this instance in static_forward_context.
        Builds cos/sin cache for position embeddings.
        """
        super().__init__(*args, **kwargs)
        # Validate supported configurations
        scaling_type = getattr(self, "scaling_type", "default")
        rope_parameters = getattr(self, "rope_parameters", {}) or {}

        is_supported = (
            scaling_type == "default"
            and "mrope_section" not in rope_parameters
            and ("use_fope" not in rope_parameters or not rope_parameters["use_fope"])
        )

        if not is_supported:
            raise NotImplementedError(
                f"SpyreRotaryEmbedding only supports default scaling without mrope_section or fope."
                f"Got scaling_type={scaling_type}, rope_parameters={rope_parameters}"
            )

        logger.debug("Building custom RotaryEmbedding")
        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16
        # Build cos/sin cache on initialization
        self._build_cos_sin_cache()
        # Compile the forward kernel
        self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)
        self._layer_name = register_layer(self, "spyre_rotary_embedding")

    def _build_cos_sin_cache(self):
        """Build cos/sin cache for position embeddings.

        Creates frequency-based position encodings:
        - frequencies = base^(-2i/rotary_dim) for i in [0, rotary_dim/2)
        - cos_cache[pos] = cos(pos * frequencies)
        - sin_cache[pos] = sin(pos * frequencies)

        Note: sin/cos calculation use float32 dtype for accuracy.
        Generated text differ from baseline when float16 used.
        """
        logger.warning_once(
            "Sin/Cos computation use float32 for accuracy.All computations are run on CPU."
        )
        compute_dtype = torch.float32

        """ Compute inverse frequencies: base^(-2i/rotary_dim)
        Using negative exponent for numerical stability"""

        i = torch.arange(0, self.rotary_dim, 2, dtype=compute_dtype)
        ratio = i / self.rotary_dim

        freq = torch.pow(self.base, ratio)

        inv_freq = 1.0 / freq

        # Create position indices [0, 1, 2, ..., max_position_embeddings-1]
        pos_id = torch.arange(self.max_position_embeddings, dtype=compute_dtype)

        # Compute frequencies for each position: pos * inv_freq
        # Shape: [max_position_embeddings, rotary_dim // 2]

        freqs = pos_id.unsqueeze(1) * inv_freq.unsqueeze(0)

        # Duplicate frequencies for interleaved pattern
        # Shape: [max_position_embeddings, rotary_dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Compute cos and sin directly in float16, then move to device
        self.cos_cache = convert(emb.cos(), None, compute_dtype)
        self.sin_cache = convert(emb.sin(), None, compute_dtype)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_rotary_embedding which retrieves this layer
        from forward_context.no_compile_layers and calls forward_impl outside
        the compilation graph.

        Args:
            positions: Position indices [batch_size, seq_len] or [total_tokens]
            query: Query tensor [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor [batch_size, seq_len, num_kv_heads, head_dim]

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs
        """
        rotated_query = torch.empty_like(query)
        rotated_key = torch.empty_like(key)

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_rotary_embedding(
            positions, query, key, rotated_query, rotated_key, self._layer_name
        )

        return rotated_query, rotated_key

    @staticmethod
    def forward_spyre(
        query: torch.Tensor,
        query_half: torch.Tensor,
        key: torch.Tensor,
        key_half: torch.Tensor,
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        cos_k: torch.Tensor,
        sin_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RoPE computation (active implementation).

        Applies rotary position embeddings to query and key tensors:
        output = x * cos + rotate_half(x) * sin

        Args:
            positions: Position indices [batch_size, seq_len] or [total_tokens]
            query: Query tensor [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor [batch_size, seq_len, num_kv_heads, head_dim]
            cos_cache: Cosine cache [max_position_embeddings, rotary_dim]
            sin_cache: Sine cache [max_position_embeddings, rotary_dim]
            rotary_dim: Dimension to apply rotation

        Returns:
            Tuple of (rotated_query, rotated_key)
        """

        query_out = query * cos_q + query_half * sin_q
        key_out = key * cos_k + key_half * sin_k

        return query_out, key_out

    def _forward_spyre_impl(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spyre device execution with device transfer and dtype conversion.

        Handles Spyre-specific constraints:
            1. Device transfer: CPU -> Spyre, convert to float16
            2. Kernel execution: Calls compiled _fwd
            3. Result transfer: Spyre -> CPU, restore original dtype

        Args:
            positions: Position indices on CPU
            query: Query tensor on CPU
            key: Key tensor on CPU

        Returns:
            Tuple of (rotated_query, rotated_key) on CPU with original dtype
        """
        query_dtype = query.dtype
        query_device = query.device
        key_dtype = key.dtype
        key_device = key.device

        # Execute compiled kernel on Spyre device

        positions = convert(positions, None, torch.int64)

        Tq, q_hidden = query.shape
        Tk, k_hidden = key.shape

        assert Tq == Tk, f"Query/Key sequence mismatch: {Tq} != {Tk}"
        T = Tq

        q_heads = q_hidden // self.head_size
        k_heads = k_hidden // self.head_size

        query = query.reshape(T, q_heads, self.head_size)
        key = key.reshape(T, k_heads, self.head_size)

        # get cos/sin
        cos = self.cos_cache[positions]  # [T, D]
        sin = self.sin_cache[positions]

        def expand_cos_sin(cos, sin, num_heads):
            cos = cos.unsqueeze(1).expand(-1, num_heads, -1)
            sin = sin.unsqueeze(1).expand(-1, num_heads, -1)
            return cos.contiguous(), sin.contiguous()

        # expand to heads
        cos_q, sin_q = expand_cos_sin(cos, sin, q_heads)
        cos_k, sin_k = expand_cos_sin(cos, sin, k_heads)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]

        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        d = self.rotary_dim // 2

        q1 = query_rot[..., :d]
        q2 = query_rot[..., d:]
        query_half = torch.cat([-q2, q1], dim=-1)

        k1 = key_rot[..., :d]
        k2 = key_rot[..., d:]
        key_half = torch.cat([-k2, k1], dim=-1)

        assert cos_q.shape == query_rot.shape, f"{cos_q.shape} != {query.shape}"
        assert sin_q.shape == query_rot.shape

        query_spyre = convert(query_rot, self._target_device, self._target_dtype)
        query_half_spyre = convert(query_half, self._target_device, self._target_dtype)

        key_spyre = convert(key_rot, self._target_device, self._target_dtype)
        key_half_spyre = convert(key_half, self._target_device, self._target_dtype)

        cos_q_spyre = convert(cos_q, self._target_device, self._target_dtype)
        sin_q_spyre = convert(sin_q, self._target_device, self._target_dtype)

        cos_k_spyre = convert(cos_k, self._target_device, self._target_dtype)
        sin_k_spyre = convert(sin_k, self._target_device, self._target_dtype)

        rotated_query, rotated_key = self.maybe_compiled_forward_spyre(
            query_spyre,
            query_half_spyre,
            key_spyre,
            key_half_spyre,
            cos_q_spyre,
            sin_q_spyre,
            cos_k_spyre,
            sin_k_spyre,
        )

        # Transfer back to CPU and restore original dtype
        rotated_query = convert(rotated_query, query_device, query_dtype)
        rotated_key = convert(rotated_key, key_device, key_dtype)

        rotated_query = torch.cat([rotated_query, query_pass], dim=-1)
        rotated_key = torch.cat([rotated_key, key_pass], dim=-1)

        rotated_query = rotated_query.reshape(T, -1)
        rotated_key = rotated_key.reshape(T, -1)

        return rotated_query, rotated_key


def _op_func(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotated_query: torch.Tensor,
    rotated_key: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    result = list(layer._forward_spyre_impl(positions, query, key))
    outputs = [rotated_query, rotated_key]
    pytree.tree_map(lambda out, res: out.copy_(res), outputs, result)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rotary_embedding custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_rotary_embedding",
        op_func=_op_func,
        mutates_args=["rotated_query", "rotated_key"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreRotaryEmbedding")
