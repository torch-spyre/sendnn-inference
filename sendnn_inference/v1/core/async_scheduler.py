# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler


class AsyncChunkedPrefillSpyreScheduler(ChunkedPrefillSpyreScheduler, AsyncScheduler):
    """Async-scheduling variant of ``ChunkedPrefillSpyreScheduler``.

    The MRO is::

        AsyncChunkedPrefillSpyreScheduler
          -> ChunkedPrefillSpyreScheduler
          -> AsyncScheduler
          -> Scheduler

    so ``super().schedule()`` and ``super().update_from_output()`` calls in
    ``ChunkedPrefillSpyreScheduler`` resolve to ``AsyncScheduler``.
    """

    pass


__all__ = [
    "AsyncChunkedPrefillSpyreScheduler",
]
