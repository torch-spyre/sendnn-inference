"""Unit tests for the pre-generated exponential-noise sampler.

These are pure CPU unit tests -- they exercise ExponentialNoisePool and the
pooled sampling path directly, without spinning up an engine.
"""

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from sendnn_inference.v1.sample.spyre_sampler import (
    ExponentialNoisePool,
    SpyreSampler,
    SpyreTopKTopPSampler,
    pooled_random_sample,
)

pytestmark = pytest.mark.cpu


def test_pool_draw_shape_dtype_and_membership():
    pool = ExponentialNoisePool(numel=10_000, dtype=torch.float32, seed=0)
    shape = torch.Size([4, 50])
    q = pool.draw(shape)

    assert q.shape == shape
    assert q.dtype == torch.float32
    # Every drawn value must come from the pool and be positive (Exp(1)).
    assert torch.all(q > 0)
    assert torch.isin(q.flatten(), pool.pool).all()


def test_pool_dtype_follows_request():
    pool = ExponentialNoisePool(numel=1024, dtype=torch.float16, seed=1)
    assert pool.draw(torch.Size([2, 8])).dtype == torch.float16


def test_pool_has_no_values_below_tiny_after_clamp():
    # exponential_() underflows to subnormal/zero values in float16 (~6e-5 of
    # samples fall below `tiny`); the clamp must lift them all to `tiny` so
    # probs.div_(q) can never divide by zero or a subnormal. This reliably
    # fails without the clamp (a 2M pool has ~120 sub-tiny values).
    tiny = torch.finfo(torch.float16).tiny
    for seed in range(4):
        pool = ExponentialNoisePool(numel=2_000_000, dtype=torch.float16, seed=seed)
        assert pool.pool.min().item() >= tiny
        assert not torch.any(pool.pool == 0)


def test_zero_noise_would_select_masked_token_but_clamp_prevents_it():
    # Demonstrates the hazard the clamp fixes: a zero in q makes 0/0 -> nan for
    # a masked (prob 0) token, and argmax then selects that masked index.
    vocab = 8
    probs = torch.zeros(1, vocab)
    probs[:, :4] = 0.25  # tokens 4..7 are masked out (prob 0)

    bad_q = torch.full((vocab,), 0.5, dtype=torch.float16)
    bad_q[5] = 0.0  # an unclamped underflow
    corrupted = probs.clone().div_(bad_q).argmax(dim=-1)
    assert corrupted.item() == 5  # masked token wrongly selected

    # The real pool is clamped at fill, so it can never contain that zero.
    pool = ExponentialNoisePool(numel=100_000, dtype=torch.float16, seed=0)
    assert pool.pool.min().item() >= torch.finfo(torch.float16).tiny


def test_pool_too_small_raises():
    pool = ExponentialNoisePool(numel=16, dtype=torch.float32, seed=0)
    with pytest.raises(ValueError, match="Noise pool too small"):
        pool.draw(torch.Size([4, 8]))  # needs 32 > 16


def test_pooled_sample_falls_back_when_batch_exceeds_pool():
    # A batch x vocab larger than the pool must not raise: it degrades to
    # upstream's on-the-fly random_sample and still returns valid tokens.
    pool = ExponentialNoisePool(numel=16, dtype=torch.float32, seed=0)
    vocab = 8
    probs = torch.full((4, vocab), 1.0 / vocab)  # 32 elements > pool of 16
    torch.manual_seed(0)
    sampled = pooled_random_sample(probs.clone(), {}, pool)
    assert sampled.shape == (4,)
    assert torch.all(sampled >= 0) and torch.all(sampled < vocab)


def test_pool_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="must be positive"):
        ExponentialNoisePool(numel=0, dtype=torch.float32)


def test_pooled_sample_peaked_probs_is_deterministic():
    # One token dominates -> argmax(probs / q) must always pick it regardless
    # of which noise slice is drawn.
    pool = ExponentialNoisePool(numel=100_000, dtype=torch.float32, seed=0)
    vocab = 32
    probs = torch.full((3, vocab), 1e-6)
    probs[:, 7] = 1.0
    for _ in range(50):
        sampled = pooled_random_sample(probs.clone(), {}, pool)
        assert torch.equal(sampled, torch.tensor([7, 7, 7]))


def test_pooled_sample_uniform_probs_covers_vocab():
    # Flat probs -> sampled token is argmin(q) over the slice, which should be
    # roughly uniform across the vocab given a large i.i.d. pool.
    pool = ExponentialNoisePool(numel=1_000_000, dtype=torch.float32, seed=0)
    vocab = 8
    probs = torch.full((1, vocab), 1.0 / vocab)
    counts = torch.zeros(vocab, dtype=torch.long)
    n_draws = 4000
    for _ in range(n_draws):
        tok = pooled_random_sample(probs.clone(), {}, pool)
        counts[tok.item()] += 1

    # Every token should be selected, and no token should dominate wildly.
    expected = n_draws / vocab
    assert torch.all(counts > 0)
    assert counts.max().item() < 3 * expected


def test_seeded_row_is_reproducible():
    # A row with a per-request generator must produce identical results across
    # runs with the same seed: the seed selects a deterministic pool offset.
    pool = ExponentialNoisePool(numel=50_000, dtype=torch.float32, seed=0)
    vocab = 64
    torch.manual_seed(123)
    probs = torch.softmax(torch.randn(2, vocab), dim=-1)

    def sample_with_seed(seed):
        gen = torch.Generator().manual_seed(seed)
        return pooled_random_sample(probs.clone(), {0: gen}, pool)

    first = sample_with_seed(999)
    second = sample_with_seed(999)
    # Only the seeded row (0) is reproducible; the unseeded row (1) correctly
    # varies as the pool offset advances between draws.
    assert first[0].item() == second[0].item()


def test_seeded_row_reads_pool_values_not_fresh_noise():
    # The seeded row's noise must come from the pool (seed selects the offset),
    # so its selected token equals argmax(probs / pool_slice) for some offset.
    pool = ExponentialNoisePool(numel=20_000, dtype=torch.float32, seed=0)
    vocab = 32
    torch.manual_seed(7)
    probs = torch.softmax(torch.randn(1, vocab), dim=-1)

    gen = torch.Generator().manual_seed(555)
    sampled = pooled_random_sample(probs.clone(), {0: gen}, pool)

    # Recompute the offset the same generator would have chosen and confirm the
    # sampled token matches a pooled slice (i.e. no fresh exponential_ noise).
    check_gen = torch.Generator().manual_seed(555)
    offset = int(torch.randint(0, pool.numel - vocab + 1, (1,), generator=check_gen).item())
    expected_q = pool.pool[offset : offset + vocab]
    expected = probs.clone().div_(expected_q).argmax(dim=-1)
    assert sampled.item() == expected.item()


def test_different_seeds_generally_differ():
    # Different seeds should pick different offsets and (usually) different
    # tokens under a flat distribution.
    pool = ExponentialNoisePool(numel=1_000_000, dtype=torch.float32, seed=0)
    vocab = 64
    probs = torch.full((1, vocab), 1.0 / vocab)
    toks = set()
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        toks.add(pooled_random_sample(probs.clone(), {0: gen}, pool).item())
    assert len(toks) > 1


def test_no_pool_matches_upstream_random_sample():
    # SpyreTopKTopPSampler with noise_pool=None must reproduce upstream's
    # PyTorch-native path bit-for-bit given the same global RNG state.
    vocab = 128
    torch.manual_seed(7)
    logits = torch.randn(4, vocab)

    sampler = SpyreTopKTopPSampler(noise_pool=None)

    torch.manual_seed(42)
    spyre_out, _ = sampler.forward_native(logits.clone(), {}, None, None)

    torch.manual_seed(42)
    probs = logits.clone().softmax(dim=-1, dtype=torch.float32)
    upstream_out = random_sample(probs, {})

    assert torch.equal(spyre_out, upstream_out)


def test_spyre_sampler_wires_pool_into_topk_sampler():
    pool = ExponentialNoisePool(numel=1024, dtype=torch.float32, seed=0)
    sampler = SpyreSampler(noise_pool=pool)
    assert isinstance(sampler.topk_topp_sampler, SpyreTopKTopPSampler)
    assert sampler.topk_topp_sampler.noise_pool is pool
    # forward must be pinned to the native (pooled) path.
    assert sampler.topk_topp_sampler.forward == sampler.topk_topp_sampler.forward_native
