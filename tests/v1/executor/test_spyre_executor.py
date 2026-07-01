"""
Tests cover:
  - execute_model: job submission, result draining, collective_rpc, SHM cleanup,
                   scheduler_output annotations, error result handling, no-op when
                   no queue is present
  - collective_rpc: triggers _try_start_mm_encoder only after warmup
  - _try_start_mm_encoder: env flag gate, startup failure → _cleanup_encoder
  - _cleanup_encoder: resets all queue/process state
"""

import queue as queue_mod
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixture: executor with parent class mocked out
# ---------------------------------------------------------------------------


@pytest.fixture
def executor():
    """SpyreMultiprocExecutor instance with MultiprocExecutor internals mocked.

    _init_executor, execute_model, collective_rpc, and shutdown are all replaced
    so no real workers are started.  The fixture yields the instance with patches
    still active so methods can be called safely during the test.
    """
    parent_execute = MagicMock(return_value=MagicMock())
    parent_collective_rpc = MagicMock(return_value=None)

    with (
        patch.object(MultiprocExecutor, "_init_executor", return_value=None),
        patch.object(MultiprocExecutor, "execute_model", parent_execute),
        patch.object(MultiprocExecutor, "collective_rpc", parent_collective_rpc),
        patch.object(MultiprocExecutor, "shutdown", return_value=None),
    ):
        exc = SpyreMultiprocExecutor.__new__(SpyreMultiprocExecutor)
        exc.vllm_config = MagicMock()
        exc._init_executor()
        # expose parent mocks for assertion
        exc._parent_execute = parent_execute
        exc._parent_collective_rpc = parent_collective_rpc
        yield exc


def _install_queues(exc, result_items=None):
    """Install mock queues on the executor.

    Uses MagicMock rather than real multiprocessing.Queue so get_nowait()
    returns preset items deterministically without IPC timing concerns.
    result_items: list of items get_nowait() should return before raising Empty.
    """
    job_q = MagicMock()
    result_q = MagicMock()
    side = list(result_items or []) + [queue_mod.Empty()]
    result_q.get_nowait.side_effect = side
    exc._mm_job_queue = job_q
    exc._mm_result_queue = result_q
    exc._mm_in_flight = 0


def _make_encode_req(req_id="req-1"):
    from sendnn_inference.v1.core.scheduler import MMEncodeRequest

    return MMEncodeRequest(request_id=req_id, prompt_token_ids=[1, 2, 3], mm_features=[])


def _make_scheduler_output(encode_reqs=None):
    out = MagicMock()
    out._spyre_mm_encode_requests = encode_reqs or []
    return out


# ---------------------------------------------------------------------------
# execute_model
# ---------------------------------------------------------------------------


class TestExecuteModel:
    def test_no_op_when_queue_not_initialised(self, executor):
        """When _mm_job_queue is None, execute_model must just delegate to super()."""
        sched = _make_scheduler_output()
        executor.execute_model(sched)
        executor._parent_execute.assert_called_once()

    def test_jobs_submitted_to_queue(self, executor):
        _install_queues(executor)
        r1 = _make_encode_req("r1")
        r2 = _make_encode_req("r2")
        sched = _make_scheduler_output([r1, r2])

        executor.execute_model(sched)

        assert executor._mm_in_flight == 2
        put_calls = executor._mm_job_queue.put_nowait.call_args_list
        assert len(put_calls) == 2
        assert put_calls[0][0][0] is r1
        assert put_calls[1][0][0] is r2

    def test_encode_requests_cleared_before_super(self, executor):
        """_spyre_mm_encode_requests must be emptied before super().execute_model."""
        _install_queues(executor)
        sched = _make_scheduler_output([_make_encode_req()])
        executor.execute_model(sched)
        assert sched._spyre_mm_encode_requests == []

    def test_successful_result_triggers_store_and_cleanup(self, executor):
        """When a result is drained, collective_rpc + cleanup must fire."""
        shape = (1, 4, 8)
        dtype = torch.float16
        _install_queues(executor, result_items=[("req-done", shape, dtype)])
        executor._mm_in_flight = 1

        sched = _make_scheduler_output()

        with patch(
            "sendnn_inference.v1.executor.spyre_executor.cleanup_embeddings_by_name"
        ) as mock_cleanup:
            executor.execute_model(sched)

        # super().collective_rpc() is called with method + forwarded kwargs
        executor._parent_collective_rpc.assert_called_once()
        rpc_call = executor._parent_collective_rpc.call_args
        assert rpc_call[0][0] == "store_mm_embeddings"
        # args is passed as a tuple wrapping the metadata list: args=([...],)
        assert rpc_call[1]["args"][0] == [("req-done", shape, dtype)]

        mock_cleanup.assert_called_once_with("req-done")
        assert sched._spyre_newly_encoded_req_ids == ["req-done"]
        assert executor._mm_in_flight == 0

    def test_error_result_sets_failed_req_ids_for_scheduler_retry(self, executor):
        """(req_id, None, None) must be collected into _spyre_failed_encode_req_ids
        so the scheduler can clear _mm_encoding_submitted and allow retry."""
        _install_queues(executor, result_items=[("req-err", None, None)])
        executor._mm_in_flight = 1

        sched = _make_scheduler_output()

        with patch("sendnn_inference.v1.executor.spyre_executor.cleanup_embeddings_by_name"):
            executor.execute_model(sched)

        # collective_rpc must NOT be called (no valid metadata)
        executor._parent_collective_rpc.assert_not_called()
        # Failed req_id must be surfaced so scheduler can clear submitted state
        assert sched._spyre_failed_encode_req_ids == ["req-err"]

    def test_in_flight_zero_skips_result_drain(self, executor):
        """When _mm_in_flight == 0, the result queue must not be polled."""
        # Even though the queue conceptually has an item, _mm_in_flight==0
        # should prevent any get_nowait() call.
        _install_queues(executor, result_items=[("req-sneaky", (1, 4, 8), torch.float16)])
        executor._mm_in_flight = 0

        sched = _make_scheduler_output()
        executor.execute_model(sched)

        executor._mm_result_queue.get_nowait.assert_not_called()
        executor._parent_collective_rpc.assert_not_called()


# ---------------------------------------------------------------------------
# collective_rpc
# ---------------------------------------------------------------------------


class TestCollectiveRpc:
    def test_triggers_encoder_start_after_warmup(self, executor):
        """collective_rpc("compile_or_warm_up_model") must call _try_start_mm_encoder."""
        with patch.object(executor, "_try_start_mm_encoder") as mock_start:
            executor.collective_rpc("compile_or_warm_up_model")
        mock_start.assert_called_once()

    def test_does_not_trigger_for_other_methods(self, executor):
        """Other collective_rpc calls must not start the encoder."""
        with patch.object(executor, "_try_start_mm_encoder") as mock_start:
            executor.collective_rpc("store_mm_embeddings", args=([],))
            executor.collective_rpc("load_model")
        mock_start.assert_not_called()

    def test_does_not_trigger_if_encoder_already_running(self, executor):
        """If _mm_encoder_proc is already set, encoder must not be re-started."""
        executor._mm_encoder_proc = MagicMock()  # simulate already running
        with patch.object(executor, "_try_start_mm_encoder") as mock_start:
            executor.collective_rpc("compile_or_warm_up_model")
        mock_start.assert_not_called()


# ---------------------------------------------------------------------------
# _try_start_mm_encoder
# ---------------------------------------------------------------------------


class TestTryStartMmEncoder:
    def test_skips_when_env_not_set(self, executor):
        """SENDNN_INFERENCE_ASYNC_MM_ENCODER=False → no process started."""
        with patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", False):
            executor._try_start_mm_encoder()

        assert executor._mm_encoder_proc is None
        assert executor._mm_job_queue is None

    def test_raises_on_startup_failure(self, executor):
        """PR #1015 finding 2: encoder startup failure must raise, not silently fall back.

        A silent fallback leaves MM scheduling permanently broken — the scheduler
        keeps gating MM requests on _mm_encoding_ready which is never populated,
        so every MM request hangs indefinitely.  Raising here lets the supervisor
        restart the process with a clear error instead of masking the failure.
        """
        fake_result_q = MagicMock()
        fake_result_q.get.return_value = "ERROR: vision load failed"

        fake_ctx = MagicMock()
        fake_ctx.Queue.side_effect = [MagicMock(), MagicMock(), fake_result_q]
        fake_ctx.Event.return_value = MagicMock()
        fake_ctx.Process.return_value = MagicMock()

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch("multiprocessing.get_context", return_value=fake_ctx),
            patch("sendnn_inference.v1.worker.mm_encoder_process.encoder_process_main"),
            pytest.raises(RuntimeError, match="Encoder process startup failed"),
        ):
            executor._try_start_mm_encoder()


# ---------------------------------------------------------------------------
# _cleanup_encoder
# ---------------------------------------------------------------------------


class TestCleanupEncoder:
    def test_resets_all_state(self, executor):
        """_cleanup_encoder must nil out all queue/process references."""
        executor._mm_stop_event = MagicMock()
        executor._mm_encoder_proc = MagicMock()
        executor._mm_encoder_proc.is_alive.return_value = False
        executor._mm_job_queue = MagicMock()
        executor._mm_result_queue = MagicMock()
        executor._mm_in_flight = 5

        executor._cleanup_encoder()

        assert executor._mm_encoder_proc is None
        assert executor._mm_job_queue is None
        assert executor._mm_result_queue is None
        assert executor._mm_stop_event is None
        assert executor._mm_in_flight == 0

    def test_terminates_live_process(self, executor):
        """A still-alive encoder process must be terminated if join times out."""
        stop = MagicMock()
        proc = MagicMock()
        proc.is_alive.side_effect = [True, True]  # alive before and after join
        executor._mm_stop_event = stop
        executor._mm_encoder_proc = proc

        executor._cleanup_encoder()

        stop.set.assert_called_once()
        proc.join.assert_called_once()
        proc.terminate.assert_called_once()
