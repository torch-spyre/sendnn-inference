# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from sendnn_inference.v1.core.scheduler import (
    ChunkedPrefillSpyreMixin,
    PoolingSpyreMixin,
)


class AsyncSpyreScheduler(AsyncScheduler):
    """Base class inheriting from the V1 async scheduler to support static
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM async scheduler
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config


class AsyncPoolingSpyreScheduler(PoolingSpyreMixin, AsyncScheduler):
    """Async scheduler for pooling models with Spyre warmup-shape constraints."""

    pass


class AsyncChunkedPrefillSpyreScheduler(ChunkedPrefillSpyreMixin, AsyncScheduler):
    """Async scheduler with Spyre chunked-prefill constraints bypassed in async mode."""

    pass


__all__ = [
    "AsyncPoolingSpyreScheduler",
    "AsyncChunkedPrefillSpyreScheduler",
]
