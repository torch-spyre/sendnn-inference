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

The fix:
  - A dedicated ``cancel_queue`` (separate from the job queue) carries
    req_id strings for aborted requests.
  - ``SpyreMultiprocExecutor`` creates the cancel queue and exposes it via
    the ``get_mm_cancel_queue()`` classmethod.
  - ``ChunkedPrefillSpyreScheduler.finish_requests`` puts the req_id on the
    cancel queue for any request that was in ``_mm_encoding_submitted``.
  - ``encoder_process_main`` accepts ``cancel_queue`` as its 5th positional
    argument and drains it before processing each job, skipping cancelled ones.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
from unittest.mock import MagicMock, patch

import pytest

from sendnn_inference.v1.core.scheduler import MMEncodeRequest

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Scheduler propagates cancellation via the cancel queue
# ---------------------------------------------------------------------------


def test_scheduler_finish_requests_puts_rid_on_cancel_queue():
    """ChunkedPrefillSpyreScheduler.finish_requests must put the req_id on the
    cancel queue for any request that is in _mm_encoding_submitted.

    The encoder drains the cancel queue before each job, so the cancelled
    request is skipped without running the expensive vision-tower forward.
    """
    from vllm.v1.request import RequestStatus

    from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler
    from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

    with patch.object(ChunkedPrefillSpyreScheduler, "__init__", lambda s, *a, **k: None):
        sched = ChunkedPrefillSpyreScheduler()

    sched._mm_encoding_submitted = {"req-aborted"}
    sched._mm_encoding_ready = set()
    sched.ongoing_prefills = []
    sched.reserved_blocks = {}
    sched.total_reserved_blocks = 0

    mock_cancel_q = MagicMock()

    with (
        patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]),
        patch.object(SpyreMultiprocExecutor, "get_mm_cancel_queue", return_value=mock_cancel_q),
    ):
        sched.finish_requests(["req-aborted"], RequestStatus.FINISHED_ABORTED)

    mock_cancel_q.put_nowait.assert_called_with("req-aborted")


# ---------------------------------------------------------------------------
# Encoder subprocess drains the cancel queue and skips cancelled jobs
# ---------------------------------------------------------------------------


def test_encoder_drains_cancel_queue_and_skips_cancelled_job():
    """encoder_process_main must accept a cancel_queue as its 5th positional
    argument, drain it before each job, and skip jobs whose req_id is present.

    execute_model must NOT be called for a job whose req_id was on the
    cancel queue before the job was dequeued.
    """
    from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

    job_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    cancel_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    # Put the cancel signal on the cancel queue BEFORE the job arrives,
    # simulating the scheduler cancelling while the encoder is busy.
    cancel_queue.put("cancelled-req")

    job = MMEncodeRequest(request_id="cancelled-req", prompt_token_ids=[1, 2, 3], mm_features=[])
    job_queue.put(job)
    job_queue.put(None)  # sentinel: terminate the loop

    mock_runner = MagicMock()

    with (
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
            return_value=mock_runner,
        ),
        patch("sendnn_inference.v1.worker.mm_encoder_process.write_embeddings"),
        patch("sendnn_inference.v1.worker.mm_encoder_process._configure_encoder_threads"),
    ):
        encoder_process_main(MagicMock(), job_queue, result_queue, stop_event, cancel_queue)

    assert result_queue.get(timeout=2) == "READY"
    # Cancelled job must have sent an abort result (req_id, None, None)
    abort_result = result_queue.get(timeout=2)
    assert abort_result == ("cancelled-req", None, None), (
        f"Expected abort result ('cancelled-req', None, None), got {abort_result}"
    )
    assert not mock_runner.execute_model.called, (
        "encoder ran execute_model on a cancelled request; "
        "this allows DoS via large-image requests that are immediately cancelled."
    )


def test_encoder_processes_job_normally_without_cancel():
    """Sanity check: when cancel_queue is empty the job is encoded normally."""
    from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

    import torch

    job_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    cancel_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    job = MMEncodeRequest(request_id="live-req", prompt_token_ids=[1, 2, 3], mm_features=[])
    job_queue.put(job)
    job_queue.put(None)

    fake_embeds = torch.zeros(1, 4, 8, dtype=torch.float16)
    mock_runner = MagicMock()
    mock_runner.execute_model.return_value = fake_embeds

    mock_shm = MagicMock()

    with (
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
            return_value=mock_runner,
        ),
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process.write_embeddings",
            return_value=mock_shm,
        ),
        patch("sendnn_inference.v1.worker.mm_encoder_process._configure_encoder_threads"),
    ):
        encoder_process_main(MagicMock(), job_queue, result_queue, stop_event, cancel_queue)

    assert result_queue.get(timeout=2) == "READY"
    req_id, shape, dtype = result_queue.get(timeout=2)
    assert req_id == "live-req"
    assert shape == (1, 4, 8)
    mock_runner.execute_model.assert_called_once()
