"""PR #1015 review — finding 1 (DoS): cancelled MM requests still consume the encoder.

When the scheduler cancels a request (client disconnect, abort, etc.) the
executor has no way to withdraw the encode job already in the queue, and the
encoder process has no way to skip a known-cancelled job before doing the
expensive `runner.execute_model(job)` call.

Failure scenario this guards against:
  - Adversary submits N requests with very large images
  - Adversary cancels all N
  - Encoder still encodes every image serially, locking up CPU/NNPA for
    minutes while no legitimate MM request can be encoded.

The fix design these tests prescribe:
  - SpyreMultiprocExecutor owns a shared `cancelled` set (Manager().dict()
    used as a set) and exposes `cancel_mm_encode(req_id)` to add to it.
  - ChunkedPrefillSpyreScheduler.finish_requests calls
    `cancel_mm_encode` on its bound executor for every aborted req_id.
  - encoder_process_main receives the shared set as its 5th positional
    argument and skips jobs whose req_id is in the set.
"""

from __future__ import annotations

import multiprocessing
from unittest.mock import MagicMock, patch

import pytest

from sendnn_inference.v1.core.scheduler import MMEncodeRequest

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Scheduler propagates cancellation to the executor
# ---------------------------------------------------------------------------


def test_scheduler_finish_requests_notifies_executor_of_cancellation():
    """ChunkedPrefillSpyreScheduler.finish_requests must, in addition to its
    local _mm_encoding_* cleanup, tell the bound executor to cancel any
    in-flight encode jobs.

    Today the scheduler only updates its own state — the encode job is
    orphaned in the queue and still consumed by the encoder.

    Note: the attribute name `_spyre_executor` here is conventional; if the
    fix wires the executor to the scheduler under a different name, adjust
    this test accordingly. The behaviour being tested is propagation, not
    the attribute name.
    """
    from vllm.v1.request import RequestStatus

    from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler

    with patch.object(ChunkedPrefillSpyreScheduler, "__init__", lambda s, *a, **k: None):
        sched = ChunkedPrefillSpyreScheduler()

    sched._mm_encoding_submitted = {"req-aborted"}
    sched._mm_encoding_ready = set()
    sched.ongoing_prefills = []
    sched.reserved_blocks = {}
    sched.total_reserved_blocks = 0

    executor = MagicMock()
    sched._spyre_executor = executor

    with patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]):
        sched.finish_requests(["req-aborted"], RequestStatus.FINISHED_ABORTED)

    assert executor.cancel_mm_encode.called, (
        "scheduler.finish_requests did not tell the executor to cancel the "
        "queued encode job for the aborted request; the encoder will still "
        "process it (DoS via cancelled large images)."
    )
    executor.cancel_mm_encode.assert_called_with("req-aborted")


# ---------------------------------------------------------------------------
# Encoder subprocess consults the shared cancellation set
# ---------------------------------------------------------------------------


def test_encoder_consults_shared_cancellation_set():
    """encoder_process_main must accept a shared cancellation set and skip
    jobs whose req_id is in it. A plain dict stands in for a Manager().dict()
    proxy — the encoder only needs `__contains__`.

    Today encoder_process_main takes four positional args and ignores any
    cancellation channel, so the call raises TypeError. After the fix, the
    call succeeds and execute_model is never invoked on the cancelled job.
    """
    from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

    job_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    cancelled = {"cancelled-req": True}  # simulated Manager().dict() proxy

    job = MMEncodeRequest(request_id="cancelled-req", prompt_token_ids=[1, 2, 3], mm_features=[])
    job_queue.put(job)
    job_queue.put(None)  # sentinel: terminate the loop

    mock_runner = MagicMock()
    mock_runner.execute_model.return_value = MagicMock()

    with (
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
            return_value=mock_runner,
        ),
        patch("sendnn_inference.v1.worker.mm_encoder_process.write_embeddings"),
        patch("sendnn_inference.v1.worker.mm_encoder_process._configure_encoder_threads"),
    ):
        try:
            encoder_process_main(MagicMock(), job_queue, result_queue, stop_event, cancelled)
        except TypeError as exc:
            pytest.fail(
                "encoder_process_main does not accept a cancellation set; "
                "without it there is no way to inform the encoder that a "
                f"queued job has been aborted. ({exc})"
            )

    assert result_queue.get(timeout=2) == "READY"
    assert not mock_runner.execute_model.called, (
        "encoder ran execute_model on a request the parent has marked "
        "cancelled. An attacker can use this to consume CPU/NNPA: submit "
        "large-image requests, cancel them immediately, and the encoder "
        "still encodes every one of them."
    )
