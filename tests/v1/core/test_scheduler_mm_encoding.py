"""
Covers:
  - Text-only requests are never gated by the MM encoding check
  - Unencoded MM requests are not promoted to prefill until encoding is ready
  - Text-only requests behind an unencoded MM request are still scheduled (no break)
  - update_from_output promotes requests from submitted → ready
  - Encoder failures abort the request immediately (no hang)
  - finish_requests cleans up encoding state
"""

import pytest
from unittest.mock import MagicMock, Mock, patch

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_request(req_id, mm_features=None, num_tokens=10):
    """Build a minimal Request for scheduler unit tests."""
    params = SamplingParams(max_tokens=20, temperature=0.0)
    req = Request(
        request_id=req_id,
        sampling_params=params,
        prompt_token_ids=list(range(num_tokens)),
        arrival_time=0,
        lora_request=None,
        pooling_params=None,
    )
    if mm_features is not None:
        req.mm_features = mm_features
    return req


@pytest.fixture
def scheduler():
    """Minimal ChunkedPrefillSpyreScheduler with heavy infra mocked out."""
    from vllm.v1.core.sched.output import CachedRequestData
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    with patch.object(ChunkedPrefillSpyreScheduler, "__init__", lambda self, *a, **k: None):
        sched = ChunkedPrefillSpyreScheduler()

    sched.waiting = FCFSRequestQueue()
    sched.skipped_waiting = FCFSRequestQueue()
    sched.running = []
    sched.ongoing_prefills = []
    sched.chunk_size = 128
    sched.do_interleaving = False
    sched.previous_step_was_prefill = False
    sched.max_num_running_reqs = 8
    sched.tkv = 0
    sched.block_size = 64
    sched.total_reserved_blocks = 0
    sched.reserved_blocks = {}
    sched.max_batch_tkv_limit = "8192"
    sched.requests = {}  # vLLM base Scheduler attribute: req_id → Request
    sched._mm_encoding_submitted = set()
    sched._mm_encoding_ready = set()
    sched._get_required_blocks = lambda req, *a, **k: (0, 0)
    sched._get_free_blocks = lambda: 100
    sched.kv_cache_manager = Mock()
    sched.kv_cache_manager.get_computed_blocks.return_value = (None, 0)
    sched.kv_cache_manager.block_pool = Mock()
    # scheduler_config needed by _current_chunk_token_threshold inside schedule()
    sched.scheduler_config = Mock()
    sched.scheduler_config.long_prefill_token_threshold = 128

    mock_output = Mock()
    mock_output.num_scheduled_tokens = {}
    mock_output.scheduled_new_reqs = []
    mock_output.scheduled_cached_reqs = CachedRequestData.make_empty()

    with (
        patch("vllm.v1.core.sched.scheduler.Scheduler.schedule", return_value=mock_output),
        patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]),
        patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output", return_value=None),
    ):
        yield sched


# ---------------------------------------------------------------------------
# can_schedule_prefill: MM encoding gate
# ---------------------------------------------------------------------------


class TestCanSchedulePrefill:
    def test_text_request_never_gated(self, scheduler):
        """Text-only request (no mm_features) must not be affected by encoding state."""
        req = _make_request("text-1")
        # Even with async encoder enabled and nothing in _mm_encoding_ready:
        # Stub remaining checks so only the MM gate matters
        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
        ):
            scheduler.running = []
            scheduler.waiting.append(Mock())  # non-empty so len check triggers
            assert scheduler.can_schedule_prefill(req) is True

    def test_mm_request_gated_when_encoding_not_ready(self, scheduler):
        """MM request must return False when async mode on and req not in _mm_encoding_ready."""
        req = _make_request("mm-1", mm_features=[Mock()])
        scheduler._mm_encoding_ready = set()

        with patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True):
            assert scheduler.can_schedule_prefill(req) is False

    def test_mm_request_passes_when_encoding_ready(self, scheduler):
        """MM request must pass the gate once its req_id is in _mm_encoding_ready."""
        req = _make_request("mm-ready", mm_features=[Mock()])
        scheduler._mm_encoding_ready = {"mm-ready"}

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
        ):
            scheduler.running = []
            assert scheduler.can_schedule_prefill(req) is True

    def test_mm_gate_inactive_without_async_encoder(self, scheduler):
        """In non-async mode the gate must not block MM requests."""
        req = _make_request("mm-sync", mm_features=[Mock()])
        scheduler._mm_encoding_ready = set()  # nothing ready

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", False),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
        ):
            scheduler.running = []
            assert scheduler.can_schedule_prefill(req) is True


# ---------------------------------------------------------------------------
# schedule(): text requests not blocked by unencoded MM at queue head
# ---------------------------------------------------------------------------


class TestScheduleMixedQueue:
    def _capture_prefill_candidates(self, scheduler, *args, **kwargs):
        """Side effect for Scheduler.schedule: capture self.waiting at call time.

        Requests in self.waiting when super().schedule() is called are the ones
        promoted for prefill this step (line ~426: self.waiting.append(new_request)).
        Requests added by the post-loop drain (line ~546) are for future steps.
        """
        self._waiting_at_super_call = list(scheduler.waiting)
        return self._mock_output

    def test_text_request_scheduled_behind_unencoded_mm(self, scheduler):
        """With an unencoded MM request at the head, a text-only request
        further back must still be promoted to prefill in the same step."""
        mm_req = _make_request("mm-not-ready", mm_features=[Mock()])
        text_req = _make_request("text-behind")

        scheduler.waiting.append(mm_req)
        scheduler.waiting.append(text_req)
        scheduler._mm_encoding_ready = set()

        self._waiting_at_super_call = []
        self._mock_output = Mock()
        self._mock_output.num_scheduled_tokens = {}
        self._mock_output.scheduled_new_reqs = []
        from vllm.v1.core.sched.output import CachedRequestData

        self._mock_output.scheduled_cached_reqs = CachedRequestData.make_empty()

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
            patch(
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                side_effect=lambda: self._capture_prefill_candidates(scheduler),
            ),
        ):
            scheduler.schedule()

        scheduled_ids = {r.request_id for r in self._waiting_at_super_call}
        assert "text-behind" in scheduled_ids, "text-only request must be promoted to prefill"
        assert "mm-not-ready" not in scheduled_ids, "unencoded MM must not be promoted this step"

    def test_unencoded_mm_not_promoted_to_waiting(self, scheduler):
        """An MM request whose encoding isn't ready must not be promoted for prefill."""
        mm_req = _make_request("mm-held", mm_features=[Mock()])
        scheduler.waiting.append(mm_req)
        scheduler._mm_encoding_ready = set()

        self._waiting_at_super_call = []
        self._mock_output = Mock()
        self._mock_output.num_scheduled_tokens = {}
        self._mock_output.scheduled_new_reqs = []
        from vllm.v1.core.sched.output import CachedRequestData

        self._mock_output.scheduled_cached_reqs = CachedRequestData.make_empty()

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
            patch(
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                side_effect=lambda: self._capture_prefill_candidates(scheduler),
            ),
        ):
            scheduler.schedule()

        scheduled_ids = {r.request_id for r in self._waiting_at_super_call}
        assert "mm-held" not in scheduled_ids

    def test_mm_encode_requests_emitted_for_unencoded_mm(self, scheduler):
        """schedule() must emit an MMEncodeRequest for each unencoded MM request."""
        mm_req = _make_request("mm-enc", mm_features=[Mock()])
        scheduler.waiting.append(mm_req)
        scheduler._mm_encoding_ready = set()
        scheduler._mm_encoding_submitted = set()

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
        ):
            output = scheduler.schedule()

        encode_reqs = getattr(output, "_spyre_mm_encode_requests", [])
        assert len(encode_reqs) == 1
        assert encode_reqs[0].request_id == "mm-enc"
        assert "mm-enc" in scheduler._mm_encoding_submitted

    def test_encode_not_re_submitted_when_already_submitted(self, scheduler):
        """A request already in _mm_encoding_submitted must not be re-emitted."""
        mm_req = _make_request("mm-dup", mm_features=[Mock()])
        scheduler.waiting.append(mm_req)
        scheduler._mm_encoding_submitted = {"mm-dup"}  # already submitted
        scheduler._mm_encoding_ready = set()

        with (
            patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
            patch.object(scheduler, "_has_scheduling_priority", return_value=True),
            patch.object(scheduler, "_satisfies_constraints", return_value=True),
        ):
            output = scheduler.schedule()

        encode_reqs = getattr(output, "_spyre_mm_encode_requests", [])
        assert len(encode_reqs) == 0


# ---------------------------------------------------------------------------
# update_from_output: encoding state transitions
# ---------------------------------------------------------------------------


class TestUpdateFromOutput:
    def _make_model_output(self):
        from sendnn_inference.v1.worker.spyre_model_runner import SpyreModelRunnerOutput

        out = SpyreModelRunnerOutput.__new__(SpyreModelRunnerOutput)
        out.req_ids = []
        out.req_id_to_index = {}
        out.sampled_token_ids = []
        out.spec_token_ids = None
        out.logprobs = None
        out.prompt_logprobs_dict = {}
        out.pooler_output = []
        out.tkv = 0
        out.left_padding = {}
        out.prefix_cache_hit_len = {}
        return out

    def _make_sched_out(self, newly_encoded=None, failed=None):
        """Build a minimal scheduler_output mock with explicit list attributes."""
        out = Mock()
        out._spyre_newly_encoded_req_ids = newly_encoded or []
        out._spyre_failed_encode_req_ids = failed or []
        return out

    def test_newly_encoded_moves_to_ready(self, scheduler):
        """_spyre_newly_encoded_req_ids must move req_ids to _mm_encoding_ready
        when the request is still live (present in scheduler.requests)."""
        scheduler._mm_encoding_submitted = {"req-A"}
        scheduler._mm_encoding_ready = set()
        scheduler.finished_req_ids = set()
        scheduler.requests["req-A"] = Mock()  # request still alive

        scheduler.update_from_output(
            self._make_sched_out(newly_encoded=["req-A"]),
            self._make_model_output(),
        )

        assert "req-A" in scheduler._mm_encoding_ready
        assert "req-A" not in scheduler._mm_encoding_submitted

    def test_newly_encoded_skips_ready_for_aborted_request(self, scheduler):
        """If a request was aborted while encoding, its late result must NOT
        add a stale entry to _mm_encoding_ready."""
        scheduler._mm_encoding_submitted = {"req-aborted"}
        scheduler._mm_encoding_ready = set()
        scheduler.finished_req_ids = set()
        # req-aborted is NOT in scheduler.requests (already removed by finish_requests)

        scheduler.update_from_output(
            self._make_sched_out(newly_encoded=["req-aborted"]),
            self._make_model_output(),
        )

        assert "req-aborted" not in scheduler._mm_encoding_ready
        assert "req-aborted" not in scheduler._mm_encoding_submitted

    def test_failed_encode_aborts_request(self, scheduler):
        """_spyre_failed_encode_req_ids must abort the request immediately."""
        scheduler._mm_encoding_submitted = {"req-fail"}
        scheduler._mm_encoding_ready = set()
        scheduler.finished_req_ids = set()

        with patch.object(scheduler, "finish_requests") as mock_finish:
            scheduler.update_from_output(
                self._make_sched_out(failed=["req-fail"]),
                self._make_model_output(),
            )

        mock_finish.assert_called_once_with(["req-fail"], RequestStatus.FINISHED_ABORTED)


# ---------------------------------------------------------------------------
# finish_requests: encoding state cleanup
# ---------------------------------------------------------------------------


class TestFinishRequests:
    def test_finish_clears_encoding_state(self, scheduler):
        """finish_requests must clean up _mm_encoding_submitted and _mm_encoding_ready."""
        scheduler._mm_encoding_submitted = {"req-1", "req-2"}
        scheduler._mm_encoding_ready = {"req-2"}

        with patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]):
            scheduler.finish_requests(["req-1", "req-2"], RequestStatus.FINISHED_STOPPED)

        assert "req-1" not in scheduler._mm_encoding_submitted
        assert "req-2" not in scheduler._mm_encoding_submitted
        assert "req-2" not in scheduler._mm_encoding_ready

    def test_finish_puts_submitted_req_on_cancel_queue(self, scheduler):
        """finish_requests must put the req_id on the cancel queue for any
        request that is in _mm_encoding_submitted so the encoder can skip it."""
        from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

        scheduler._mm_encoding_submitted = {"req-cancel"}
        scheduler._mm_encoding_ready = set()

        mock_cq = MagicMock()
        with (
            patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]),
            patch.object(SpyreMultiprocExecutor, "get_mm_cancel_queue", return_value=mock_cq),
        ):
            scheduler.finish_requests(["req-cancel"], RequestStatus.FINISHED_ABORTED)

        mock_cq.put_nowait.assert_called_with("req-cancel")
        assert "req-cancel" not in scheduler._mm_encoding_submitted

    def test_scheduler_finish_requests_notifies_executor_of_cancellation(self, scheduler):
        """ChunkedPrefillSpyreScheduler.finish_requests must, in addition to its
        local _mm_encoding_* cleanup, tell the bound executor to cancel any
        in-flight encode jobs.

        Without this, the encode job is orphaned in the queue and still consumed
        by the encoder — wasting CPU/NNPA on a dead request (DoS vector for large
        images: submit N, cancel N, encoder encodes all N serially).

        The notification is sent via the cancel queue exposed by
        SpyreMultiprocExecutor.get_mm_cancel_queue(); the encoder subprocess drains
        it before each job and skips any req_id it finds there.
        """
        from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

        scheduler._mm_encoding_submitted = {"req-aborted"}
        scheduler._mm_encoding_ready = set()

        mock_cq = MagicMock()
        with (
            patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]),
            patch.object(SpyreMultiprocExecutor, "get_mm_cancel_queue", return_value=mock_cq),
        ):
            scheduler.finish_requests(["req-aborted"], RequestStatus.FINISHED_ABORTED)

        assert mock_cq.put_nowait.called, (
            "scheduler.finish_requests did not notify the encoder of the cancelled "
            "request; the encoder will still process it (DoS via cancelled large images)."
        )
        mock_cq.put_nowait.assert_called_with("req-aborted")

    def test_finish_all_clears_everything(self, scheduler):
        """finish_requests(None, …) must clear all encoding sets."""
        scheduler._mm_encoding_submitted = {"req-A", "req-B"}
        scheduler._mm_encoding_ready = {"req-C"}

        with patch("vllm.v1.core.sched.scheduler.Scheduler.finish_requests", return_value=[]):
            scheduler.finish_requests(None, RequestStatus.FINISHED_STOPPED)

        assert len(scheduler._mm_encoding_submitted) == 0
        assert len(scheduler._mm_encoding_ready) == 0
