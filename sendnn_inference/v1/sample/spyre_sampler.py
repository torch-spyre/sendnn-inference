"""Spyre sampler with a pre-generated exponential-noise pool.

Random number generation via ``Tensor.exponential_()`` is slow on s390x, and
vLLM's ``random_sample`` calls it once per decode step to draw the Gumbel-max
noise ``q`` used for weighted sampling (``argmax(probs / q)``).

To avoid that per-step cost, ``ExponentialNoisePool`` generates a large buffer
of i.i.d. Exp(1) values a single time at model load and, on each step, returns
a fresh random slice of it. A contiguous slice of an i.i.d. pool is itself a
valid i.i.d. draw, so sampling stays statistically sound; the random offset
decorrelates successive steps. Per-request seeded generators pick their pool
offset *from the seed* (see ``pooled_random_sample``) so reproducible requests
stay reproducible while also skipping the slow per-step generation.

This is opt-in via ``SENDNN_INFERENCE_USE_NOISE_POOL``; when disabled the
classes below behave identically to upstream vLLM.
"""

import random
import time

import torch
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    random_sample,
)
from vllm.v1.sample.sampler import Sampler

import sendnn_inference.envs as envs_spyre

logger = init_logger(__name__)


class ExponentialNoisePool:
    """A fixed buffer of Exp(1) noise generated once, sliced at random per draw.

    The pool is filled a single time at construction (the one slow
    ``exponential_()`` call, paid at model load). ``draw`` then returns a
    random contiguous view of the requested shape with no further RNG on the
    hot path beyond picking a single integer offset.
    """

    def __init__(
        self,
        numel: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if numel <= 0:
            raise ValueError(f"Noise pool size must be positive, got {numel}")
        self.numel = numel
        self.dtype = dtype
        self.device = torch.device(device)
        # Deterministic pool so runs are reproducible given the same seed; the
        # per-step offset uses Python's (fast, non-torch) RNG so we never touch
        # the slow s390x torch RNG on the hot path.
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        self.pool = torch.empty(numel, dtype=dtype, device=self.device)
        self.pool.exponential_(generator=gen)
        # exponential_() can underflow to exactly 0.0 in low-precision dtypes
        # (float16 hits it at ~6e-8), which would make probs.div_(q) produce
        # +inf/nan and select a wrong or top-k/top-p-masked token. Clamp to the
        # smallest positive normal so the pool never contains a zero. This is a
        # no-op in practice for float32 (its tiny is ~1e-38).
        self.pool.clamp_(min=torch.finfo(dtype).tiny)
        self._offset_rng = random.Random(seed)

    def draw(self, shape: torch.Size) -> torch.Tensor:
        """Return a fresh Exp(1) tensor of ``shape`` sliced from the pool."""
        n = 1
        for dim in shape:
            n *= dim
        if n > self.numel:
            raise ValueError(
                f"Noise pool too small: need {n} elements for shape {tuple(shape)} "
                f"but pool holds {self.numel}. Increase "
                f"SENDNN_INFERENCE_NOISE_POOL_MULTIPLIER."
            )
        offset = self._offset_rng.randint(0, self.numel - n)
        logger.debug(
            "Noise pool draw: shape=%s (%d elems) from offset %d/%d",
            tuple(shape),
            n,
            offset,
            self.numel,
        )
        return self.pool[offset : offset + n].view(shape)

    def draw_row(self, width: int, generator: torch.Generator) -> torch.Tensor:
        """Return one ``width``-element row whose offset is chosen by ``generator``.

        Used for seeded requests: the per-request generator deterministically
        selects *where* in the pool to read, so the same seed yields the same
        noise (reproducible) without generating any new random values in bulk.
        The generator advances by a single int per call, so successive decode
        steps for the same request read different slices.
        """
        if width > self.numel:
            raise ValueError(
                f"Noise pool too small: need {width} elements for a row but pool "
                f"holds {self.numel}. Increase SENDNN_INFERENCE_NOISE_POOL_MULTIPLIER."
            )
        offset = int(
            torch.randint(
                0, self.numel - width + 1, (1,), generator=generator, device=self.device
            ).item()
        )
        return self.pool[offset : offset + width]


def pooled_random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    pool: ExponentialNoisePool,
) -> torch.Tensor:
    """Drop-in for vLLM ``random_sample`` backed by ``pool``.

    Seeded requests use their per-request generator to *select an offset* into
    the pool (via ``pool.draw_row``) rather than generating fresh noise. This
    keeps their sampling reproducible for a given seed while still skipping the
    slow per-step ``exponential_()``. Unseeded rows read a random pool slice.

    Note: because seeded rows now read pool values instead of freshly generated
    ones, seeded outputs are reproducible run-to-run (given the same pool) but
    are NOT bit-identical to upstream vLLM's seeded outputs.

    If the requested batch x vocab exceeds the pool capacity, this falls back
    to upstream's on-the-fly ``random_sample`` for that step rather than
    failing -- so an unexpectedly large batch degrades to the (slower) fresh
    noise path instead of erroring.
    """
    n = probs.shape.numel()
    if n > pool.numel:
        logger.warning_once(
            "Sampling batch needs %d noise elements but the pool holds only %d; "
            "falling back to on-the-fly exponential_() for oversized steps. "
            "Increase SENDNN_INFERENCE_NOISE_POOL_MULTIPLIER to avoid this.",
            n,
            pool.numel,
        )
        return random_sample(probs, generators)
    q = pool.draw(probs.shape)
    if generators:
        # draw() returns a view into the shared pool; clone before overwriting
        # seeded rows so we never mutate the pool buffer itself.
        q = q.clone()
        width = probs.shape[1]
        for i, generator in generators.items():
            q[i] = pool.draw_row(width, generator)
        logger.debug(
            "Pooled sample: batch=%d, %d seeded row(s) via seed-selected offset",
            probs.shape[0],
            len(generators),
        )
    else:
        logger.debug("Pooled sample: batch=%d, all rows from pool", probs.shape[0])
    # Mirror upstream random_sample's Gumbel-max trick. probs is a fresh
    # softmax output, so the in-place div is safe; q (a pool view when no
    # seeded rows) is only read.
    return probs.div_(q).argmax(dim=-1).view(-1)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """``TopKTopPSampler`` that draws sampling noise from a pre-generated pool.

    When ``noise_pool`` is ``None`` this is equivalent to the upstream
    PyTorch-native path.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        noise_pool: ExponentialNoisePool | None = None,
    ) -> None:
        super().__init__(logprobs_mode)
        self.noise_pool = noise_pool
        # The parent picks a forward_* implementation based on current_platform;
        # pin ours so the pool path is used deterministically on Spyre.
        self.forward = self.forward_native

        # Opt-in latency instrumentation for the noise + sampling step.
        self._timing_interval = envs_spyre.SENDNN_INFERENCE_SAMPLER_TIMING
        self._timing_enabled = self._timing_interval > 0
        self._timing_calls = 0
        self._timing_total_s = 0.0

        logger.info(
            "SpyreTopKTopPSampler initialized: noise_pool=%s, timing=%s",
            "on" if self.noise_pool is not None else "off",
            f"every {self._timing_interval} calls" if self._timing_enabled else "off",
        )

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        start = time.perf_counter() if self._timing_enabled else 0.0
        if self.noise_pool is None:
            sampled = random_sample(probs, generators)
        else:
            sampled = pooled_random_sample(probs, generators, self.noise_pool)
        if self._timing_enabled:
            self._record_timing(time.perf_counter() - start, probs.shape)
        return sampled, logits_to_return

    def _record_timing(self, elapsed_s: float, shape: torch.Size) -> None:
        """Accumulate sampling latency and log a running average periodically."""
        self._timing_calls += 1
        self._timing_total_s += elapsed_s
        if self._timing_calls >= self._timing_interval:
            path = "pool" if self.noise_pool is not None else "exponential_"
            logger.info(
                "Spyre sampler [%s]: %.3f ms/call avg over %d calls "
                "(last shape: batch=%d, vocab=%d)",
                path,
                1000.0 * self._timing_total_s / self._timing_calls,
                self._timing_calls,
                shape[0],
                shape[1],
            )
            self._timing_calls = 0
            self._timing_total_s = 0.0


class SpyreSampler(Sampler):
    """``Sampler`` that routes weighted sampling through ``SpyreTopKTopPSampler``.

    All other behaviour (penalties, temperature, greedy path, logprobs) is
    inherited unchanged from upstream.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        noise_pool: ExponentialNoisePool | None = None,
    ) -> None:
        super().__init__(logprobs_mode)
        self.topk_topp_sampler = SpyreTopKTopPSampler(logprobs_mode, noise_pool=noise_pool)
