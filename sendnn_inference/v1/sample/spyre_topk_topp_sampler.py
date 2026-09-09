# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

import torch
from vllm.logger import init_logger
from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, TopKTopPSampler

from sendnn_inference.v1.sample.async_ring_buffer import AsyncExponential_RingBuffer

logger = init_logger(__name__)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """Top-k/top-p sampler optimized for Spyre hardware via asynchronous noise pre-sampling.

    This removes CPU-bound noise generation from the latency-critical sampling path by
    pre-drawing noise into a ring buffer that the decoder can consume via zero-copy views
    during token selection. The buffer is pre-allocated based on vocab_size and multiples
    of max_batch_size to support zero-copy access patterns.
    """

    def __init__(
        self,
        vocab_size: int,
        max_batch_size: int,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ):
        """Initialize the SpyreTopKTopPSampler with a asynchronous exponential
        noise ring buffer.

        Args:
            vocab_size: The size of the vocabulary (number of possible tokens).
                Used to allocate noise buffer rows of appropriate size.
            max_batch_size: The maximum batch size that will be processed.
                Determines the total capacity of the pre-allocated noise buffer.
            logprobs_mode: See vllm.v1.sample.ops.topk_topp_sampler for details.
        """
        super().__init__(logprobs_mode=logprobs_mode)

        self._noise_buffer = AsyncExponential_RingBuffer(
            vocab_size=vocab_size,
            max_batch_size=max_batch_size,
        )

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply top-k/top-p filtering and sample tokens using pre-drawn noise."""

        if generators:
            logger.warning_once(
                "Generators are not supported by SpyreTopKTopPSampler. Falling back to base class.",
            )
            return super().forward_native(logits, generators, k, p)

        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        with self._noise_buffer.borrow_rows(n=logits.shape[0]) as log_noise:
            sample_result = SpyreTopKTopPSampler._sample_with_predrawn_log_noise(logits, log_noise)

        return sample_result, logits_to_return

    def shutdown(self) -> None:
        """Shutdown the sampler and clean up resources."""
        self._noise_buffer.shutdown()

    @staticmethod
    def _sample_with_predrawn_noise(probs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample using pre-drawn exponential noise (no exponential_() call)."""
        return probs.div(noise).argmax(dim=-1).view(-1)

    @staticmethod
    def _sample_with_predrawn_log_noise(
        logits: torch.Tensor, log_noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample using pre-drawn exponential log noise (no exponential_() call)."""
        return (logits - log_noise).argmax(dim=-1).view(-1)
