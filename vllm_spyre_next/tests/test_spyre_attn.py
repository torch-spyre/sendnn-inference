# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm_spyre_next.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
)


def is_spyre_available():
    try:
        test_tensor = torch.randn(1, device=torch.device("spyre"))
        del test_tensor
        return True
    except Exception:
        return False


SPYRE_AVAILABLE = is_spyre_available()

pytestmark = pytest.mark.skipif(
    not SPYRE_AVAILABLE, reason="Spyre device not available - these tests require Spyre hardware"
)

NUM_HEADS = [(4, 4), (8, 2)]  # (num_query_heads, num_kv_heads)
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16]
NUM_BLOCKS = [2048, 32768]


def ref_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    """Reference implementation of attention for validation."""
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale  # avoid in-place mutation of the input tensor

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1)
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 256), (2, 128), (4, 512)],
        [(1, 256), (1, 128), (1, 512)],
        [(72, 512), (1, 256), (4, 128)],
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("use_sdpa", [True, False])
@torch.inference_mode()
def test_spyre_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    use_sdpa: bool,
) -> None:
    """Validate SpyreAttentionImpl against a reference implementation."""
    torch.set_default_device("cpu")
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)

    kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # Pre-populate KV cache with historical context
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        historical_len = kv_len - query_len
        if historical_len > 0:
            historical_keys = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            historical_values = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            for token_idx in range(historical_len):
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                actual_block = block_tables[seq_idx, block_idx].item()
                key_cache[actual_block, block_offset] = historical_keys[token_idx]
                value_cache[actual_block, block_offset] = historical_values[token_idx]

    # Create slot mapping for new query tokens
    slot_mapping = []
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        for token_idx in range(query_len):
            pos = kv_len - query_len + token_idx
            actual_block = block_tables[seq_idx, pos // block_size].item()
            slot_mapping.append(actual_block * block_size + pos % block_size)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

    attn_metadata = SpyreAttentionMetadata(
        num_actual_tokens=sum(query_lens),
        num_seqs=num_seqs,
        max_query_len=max_query_len,
        max_seq_len=max_kv_len,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        block_size=block_size,
        slot_mapping=slot_mapping,
        apply_causal_mask=max_query_len > 1,
        num_kv_heads=num_kv_heads,
        num_heads=num_query_heads,
    )

    attn_impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=soft_cap,
        use_sdpa=use_sdpa,
    )

    output = torch.empty_like(query)
    attn_impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    ref_output = ref_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    if use_sdpa:
        atol, rtol = 0.1, 0.1
    elif max_query_len >= 32:
        atol, rtol = 0.3, 5.0  # float16 accumulation errors for large prompts
    else:
        atol, rtol = 0.2, 0.2

    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16])
@torch.inference_mode()
def test_spyre_attn_single_sequence(
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """Test single-sequence attention across a range of query/kv lengths."""
    torch.set_default_device("cpu")
    set_random_seed(42)

    num_query_heads, num_kv_heads = num_heads
    scale = head_size**-0.5

    test_cases = [
        (1, 128),  # single token decode
        (32, 256),  # exact chunk size
        (64, 512),  # multi-chunk
        (100, 512),  # non-aligned query length
    ]

    for query_len, kv_len in test_cases:
        num_seqs = 1
        num_blocks = 1024

        query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
        key = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype)
        value = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype)

        kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]

        max_num_blocks_per_seq = (kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
        )

        historical_len = kv_len - query_len
        if historical_len > 0:
            historical_keys = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            historical_values = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            for token_idx in range(historical_len):
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                actual_block = block_tables[0, block_idx].item()
                key_cache[actual_block, block_offset] = historical_keys[token_idx]
                value_cache[actual_block, block_offset] = historical_values[token_idx]

        slot_mapping = []
        for token_idx in range(query_len):
            pos = kv_len - query_len + token_idx
            actual_block = block_tables[0, pos // block_size].item()
            slot_mapping.append(actual_block * block_size + pos % block_size)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

        attn_metadata = SpyreAttentionMetadata(
            num_actual_tokens=query_len,
            num_seqs=num_seqs,
            max_query_len=query_len,
            max_seq_len=kv_len,
            seq_lens=torch.tensor([kv_len], dtype=torch.int32),
            query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
            block_table=block_tables,
            block_size=block_size,
            slot_mapping=slot_mapping,
            apply_causal_mask=query_len > 1,
            num_kv_heads=num_kv_heads,
            num_heads=num_query_heads,
        )

        attn_impl = SpyreAttentionImpl(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
        )

        output = torch.empty_like(query)
        attn_impl.forward(
            layer=None,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

        ref_output = ref_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=[query_len],
            kv_lens=[kv_len],
            block_tables=block_tables,
            scale=scale,
        )

        if query_len >= 32:
            atol, rtol = 0.3, 5.0  # float16 accumulation errors for large prompts
        else:
            atol, rtol = 0.1, 0.1

        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
