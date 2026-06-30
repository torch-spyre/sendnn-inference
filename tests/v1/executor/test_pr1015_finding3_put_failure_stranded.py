"""PR #1015 review — finding 3 (stuck request): put_nowait failure silently
swallowed; scheduler hangs the request forever.

`SpyreMultiprocExecutor.execute_model` submits encode jobs with:

    try:
        self._mm_job_queue.put_nowait(req)
        self._mm_in_flight += 1
    except Exception as exc:
        logger.debug("MM job queue error for req '%s': %s", req.request_id, exc)

The scheduler has ALREADY added the req_id to `_mm_encoding_submitted` by
the time the executor sees the request. After put_nowait raises, that set
membership prevents the scheduler from ever re-emitting an encode job for
the same id. The request is permanently stranded with no result arriving
and no error surfaced to the client.

Triggers in real deployments:
  - BrokenPipeError when the encoder subprocess has died.
  - queue.Full once the queue is bounded (defense-in-depth for a dead
    encoder).
  - PicklingError on an exotic mm_features payload.

These tests fail today and will pass once `execute_model` records
put-failures on `scheduler_output._spyre_failed_encode_req_ids`. The
existing `test_failed_encode_aborts_request` in test_scheduler_mm_encoding.py
already covers what the scheduler does with that list — we just need the
executor to populate it.
"""

from __future__ import annotations

import queue as queue_mod
from unittest.mock import MagicMock, patch

import pytest

from sendnn_inference.v1.core.scheduler import MMEncodeRequest

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixture: real executor with parent class mocked, real attribute layout
# ---------------------------------------------------------------------------


@pytest.fixture
def executor():
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

    with (
        patch.object(MultiprocExecutor, "_init_executor", return_value=None),
        patch.object(MultiprocExecutor, "execute_model", MagicMock(return_value=MagicMock())),
        patch.object(MultiprocExecutor, "collective_rpc", MagicMock(return_value=None)),
        patch.object(MultiprocExecutor, "shutdown", return_value=None),
    ):
        exc = SpyreMultiprocExecutor.__new__(SpyreMultiprocExecutor)
        exc.vllm_config = MagicMock()
        exc._init_executor()
        yield exc


def _empty_result_queue():
    q = MagicMock()
    q.get_nowait.side_effect = queue_mod.Empty()
    return q


def _scheduler_output(encode_reqs):
    """Build a plain object so we can assert attribute writes (MagicMock
    accepts every attribute and would mask the bug)."""

    class _SO:
        pass

    so = _SO()
    so._spyre_mm_encode_requests = encode_reqs
    so.finished_req_ids = []
    return so


# ---------------------------------------------------------------------------
# BrokenPipeError (encoder process died) must be surfaced
# ---------------------------------------------------------------------------


def test_broken_pipe_during_put_surfaces_to_scheduler(executor):
    """When the encoder subprocess has died, put_nowait raises BrokenPipeError.
    The executor MUST add the req_id to `_spyre_failed_encode_req_ids` so
    the scheduler can clean up its `_mm_encoding_submitted` state and abort
    the request. Today it just logs at DEBUG and moves on.
    """
    executor._mm_job_queue = MagicMock()
    executor._mm_job_queue.put_nowait.side_effect = BrokenPipeError("encoder died")
    executor._mm_result_queue = _empty_result_queue()
    executor._mm_in_flight = 0

    req = MMEncodeRequest(request_id="stranded-bp", prompt_token_ids=[1, 2], mm_features=[])
    sched_out = _scheduler_output([req])

    executor.execute_model(sched_out)

    failed = getattr(sched_out, "_spyre_failed_encode_req_ids", [])
    assert "stranded-bp" in failed, (
        "put_nowait raised BrokenPipeError but the executor did not record "
        "the req_id on _spyre_failed_encode_req_ids; the scheduler keeps "
        "'stranded-bp' in _mm_encoding_submitted forever and the request "
        "hangs."
    )


# ---------------------------------------------------------------------------
# queue.Full must be surfaced (forward-compat with bounded queue)
# ---------------------------------------------------------------------------


def test_queue_full_during_put_surfaces_to_scheduler(executor):
    """Defense-in-depth: once the encoder subprocess dies, multiprocessing.Queue
    will eventually start raising queue.Full (the parent's feeder buffer
    fills because nothing consumes it). The executor must surface this the
    same way it surfaces BrokenPipeError — otherwise the silent stranding
    becomes the dominant failure mode for a dead encoder.
    """
    executor._mm_job_queue = MagicMock()
    executor._mm_job_queue.put_nowait.side_effect = queue_mod.Full()
    executor._mm_result_queue = _empty_result_queue()
    executor._mm_in_flight = 0

    req = MMEncodeRequest(request_id="stranded-full", prompt_token_ids=[1, 2], mm_features=[])
    sched_out = _scheduler_output([req])

    executor.execute_model(sched_out)

    failed = getattr(sched_out, "_spyre_failed_encode_req_ids", [])
    assert "stranded-full" in failed, (
        "put_nowait raised queue.Full but the executor did not record the "
        "req_id on _spyre_failed_encode_req_ids."
    )


# ---------------------------------------------------------------------------
# _mm_in_flight must not be incremented when put_nowait raised
# ---------------------------------------------------------------------------


def test_in_flight_not_incremented_on_put_failure(executor):
    """Regression guard: the bookkeeping counter must stay consistent with
    what is actually in the queue. If put_nowait raised, the job is NOT in
    flight and the counter must stay where it was.

    Today's behaviour happens to be correct (the increment follows the
    call), but the wider fix may rewrite this region — lock it in.
    """
    executor._mm_job_queue = MagicMock()
    executor._mm_job_queue.put_nowait.side_effect = BrokenPipeError("encoder died")
    executor._mm_result_queue = _empty_result_queue()
    executor._mm_in_flight = 0

    req = MMEncodeRequest(request_id="bookkeeping", prompt_token_ids=[1], mm_features=[])
    sched_out = _scheduler_output([req])

    executor.execute_model(sched_out)

    assert executor._mm_in_flight == 0, (
        "_mm_in_flight was incremented even though put_nowait raised; "
        "the result-queue drain loop will be triggered on a phantom job "
        "and never settle."
    )
