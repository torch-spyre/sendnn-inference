# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler


class AsyncChunkedPrefillSpyreScheduler(ChunkedPrefillSpyreScheduler, AsyncScheduler):
    """Async-scheduling variant of ``ChunkedPrefillSpyreScheduler``.

    ``ChunkedPrefillSpyreScheduler`` and ``AsyncScheduler`` both subclass
    ``Scheduler``. For this subclass (only), C3 linearization places
    ``ChunkedPrefillSpyreScheduler`` before ``AsyncScheduler`` before the
    shared ``Scheduler`` base, so inside ``ChunkedPrefillSpyreScheduler``
    methods ``super()`` resolves to ``AsyncScheduler``, not ``Scheduler``.
    """

    pass


__all__ = [
    "AsyncChunkedPrefillSpyreScheduler",
]
