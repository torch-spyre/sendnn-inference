"""Unit tests for async scheduler mixin classes.

Tests the PoolingSpyreMixin and ChunkedPrefillSpyreMixin behaviour when used
with AsyncScheduler (async mode) vs Scheduler (sync mode).

Key behaviours under test:
  - _is_async_scheduler() correctly identifies async vs sync instances
  - PoolingSpyreMixin.schedule() applies warmup-shape constraints in both modes
  - ChunkedPrefillSpyreMixin.schedule() applies the same Spyre constraints in both
    modes (one prefill at a time, no mixing prefill and decode); the only async-
    specific step is clearing stale ongoing_prefills entries that were speculatively
    marked complete by _update_after_schedule before update_from_output() confirmed them
  - ChunkedPrefillSpyreMixin.update_from_output() filters scheduler output in async
    mode to only include requests that were actually executed that step
"""

import pytest
from collections import deque
from unittest.mock import Mock, patch

from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request

from sendnn_inference.v1.core.async_scheduler import (
    AsyncChunkedPrefillSpyreScheduler,
    AsyncPoolingSpyreScheduler,
)
from sendnn_inference.v1.core.scheduler import (
    ChunkedPrefillSpyreScheduler,
    PoolingSpyreScheduler,
)
from sendnn_inference.v1.worker.spyre_model_runner import SpyreModelRunnerOutput

pytestmark = pytest.mark.skip_global_cleanup


def _make_request(request_id: str, prompt_len: int = 50, max_tokens: int = 20) -> Request:
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    return Request(
        request_id=request_id,
        sampling_params=sampling_params,
        prompt_token_ids=list(range(prompt_len)),
        arrival_time=0,
        lora_request=None,
        pooling_params=None,
    )


def _make_scheduler_output(req_ids: list[str], tokens_per_req: int = 10) -> SchedulerOutput:
    num_scheduled_tokens = {req_id: tokens_per_req for req_id in req_ids}
    cached_reqs = CachedRequestData(
        req_ids=[],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=tokens_per_req * len(req_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
    )


def _make_model_runner_output(req_ids: list[str], tkv: int = 10) -> SpyreModelRunnerOutput:
    return SpyreModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[1]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        left_padding={},
        prefix_cache_hit_len={},
        tkv=tkv,
    )


def _patch_init(cls):
    """Context manager: bypass __init__ for a scheduler class."""
    return patch.object(cls, "__init__", lambda self, *a, **kw: None)


def _set_common_attrs(scheduler, chunk_size=128, max_seqs=4):
    """Set the minimum set of attributes needed by the mixin methods."""
    mock_vllm_config = Mock()
    mock_vllm_config.model_config.max_model_len = 2048
    mock_vllm_config.scheduler_config.max_num_batched_tokens = chunk_size
    mock_vllm_config.scheduler_config.max_num_seqs = max_seqs

    scheduler.vllm_config = mock_vllm_config
    scheduler.model_config = mock_vllm_config.model_config
    scheduler.scheduler_config = mock_vllm_config.scheduler_config
    scheduler.waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler.max_num_running_reqs = max_seqs


def _set_chunked_prefill_attrs(scheduler, chunk_size=128):
    """Set extra attributes required by ChunkedPrefillSpyreMixin."""
    scheduler.chunk_size = chunk_size
    scheduler.ongoing_prefills = []
    scheduler.do_interleaving = False
    scheduler.previous_step_was_prefill = False
    scheduler.tkv = 0
    scheduler.block_size = 64
    scheduler.max_batch_tkv_limit = 131072
    # Run-ahead snapshot attributes (see ChunkedPrefillSpyreMixin.__init__)
    scheduler._prefill_scheduled_num_computed_queue = deque()
    scheduler._scheduled_ongoing_prefill_ids = frozenset()
    # True = "schedule() was called, commit pending"; gates the run-ahead guard.
    # Tests that simulate the run-ahead scenario (e.g. speculative completion)
    # need this True so the guard can fire and restore ongoing_prefills.
    scheduler._schedule_awaiting_commit = True
    # skipped_waiting holds grammar-blocked requests in the base vLLM Scheduler
    scheduler.skipped_waiting = FCFSRequestQueue()


def _set_pooling_attrs(scheduler):
    """Set extra attributes required by PoolingSpyreMixin."""
    scheduler.spyre_warmup_shapes = (
        {"prompt_length": 64, "batch_size": 4},
        {"prompt_length": 128, "batch_size": 2},
    )


class TestIsAsyncScheduler:
    """Verify that _is_async_scheduler() correctly detects the concrete base."""

    def test_sync_pooling_is_not_async(self):
        with _patch_init(PoolingSpyreScheduler):
            s = PoolingSpyreScheduler()
        assert s._is_async_scheduler() is False

    def test_sync_chunked_prefill_is_not_async(self):
        with _patch_init(ChunkedPrefillSpyreScheduler):
            s = ChunkedPrefillSpyreScheduler()
        assert s._is_async_scheduler() is False

    def test_async_pooling_is_async(self):
        with _patch_init(AsyncPoolingSpyreScheduler):
            s = AsyncPoolingSpyreScheduler()
        assert s._is_async_scheduler() is True

    def test_async_chunked_prefill_is_async(self):
        with _patch_init(AsyncChunkedPrefillSpyreScheduler):
            s = AsyncChunkedPrefillSpyreScheduler()
        assert s._is_async_scheduler() is True


class TestPoolingSpyreMixinSchedule:
    """PoolingSpyreMixin applies warmup-shape constraints in both sync and async modes."""

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(PoolingSpyreScheduler):
            s = PoolingSpyreScheduler()
        _set_common_attrs(s)
        _set_pooling_attrs(s)
        return s

    @pytest.fixture
    def async_scheduler(self):
        with _patch_init(AsyncPoolingSpyreScheduler):
            s = AsyncPoolingSpyreScheduler()
        _set_common_attrs(s)
        _set_pooling_attrs(s)
        return s

    def _run_schedule(self, scheduler):
        """Patch the base scheduler.schedule() and call our mixin.

        The fake base drains ``waiting`` into ``running`` to simulate what the
        real Scheduler does, so tests can assert on post-schedule waiting state.
        """
        mock_output = Mock(spec=SchedulerOutput)
        mock_output.has_structured_output_requests = False
        base_path = "vllm.v1.core.sched.scheduler.Scheduler.schedule"

        def _fake_base(self_inner):
            while self_inner.waiting:
                self_inner.running.append(self_inner.waiting.popleft())
            return mock_output

        with patch(base_path, _fake_base):
            return scheduler.schedule(), mock_output

    def test_sync_applies_warmup_shape_constraints(self, sync_scheduler):
        """Requests that exceed warmup shapes are held back."""
        # prompt_len=200 exceeds the max shape prompt_length=128
        req_too_long = _make_request("too_long", prompt_len=200)
        req_fits = _make_request("fits", prompt_len=50)

        sync_scheduler.waiting.append(req_too_long)
        sync_scheduler.waiting.append(req_fits)

        output, _ = self._run_schedule(sync_scheduler)

        # req_too_long should remain in waiting (held back)
        waiting_ids = [r.request_id for r in sync_scheduler.waiting]
        assert "too_long" in waiting_ids
        # req_fits should have been released to the base scheduler
        assert "fits" not in waiting_ids

    def test_async_applies_warmup_shape_constraints(self, async_scheduler):
        """Same warmup-shape constraints apply in async mode."""
        req_too_long = _make_request("too_long", prompt_len=200)
        req_fits = _make_request("fits", prompt_len=50)

        async_scheduler.waiting.append(req_too_long)
        async_scheduler.waiting.append(req_fits)

        # AsyncScheduler.schedule() is also patched via the Scheduler base;
        # the fake drains waiting to simulate real scheduling.
        base_path = "vllm.v1.core.sched.scheduler.Scheduler.schedule"
        mock_output = Mock(spec=SchedulerOutput)
        mock_output.has_structured_output_requests = False

        def _fake_base(self_inner):
            while self_inner.waiting:
                self_inner.running.append(self_inner.waiting.popleft())
            return mock_output

        with patch(base_path, _fake_base):
            async_scheduler.schedule()

        waiting_ids = [r.request_id for r in async_scheduler.waiting]
        assert "too_long" in waiting_ids
        assert "fits" not in waiting_ids

    def test_held_back_requests_restored_after_schedule(self, sync_scheduler):
        """All held-back requests are restored to waiting after schedule()."""
        req_too_long = _make_request("too_long", prompt_len=200)
        sync_scheduler.waiting.append(req_too_long)

        self._run_schedule(sync_scheduler)

        waiting_ids = [r.request_id for r in sync_scheduler.waiting]
        assert "too_long" in waiting_ids

    def test_running_batch_skips_warmup_shape_filter(self, sync_scheduler):
        """When requests are already running, waiting queue is held back entirely."""
        running_req = _make_request("running", prompt_len=50)
        sync_scheduler.running = [running_req]

        new_req = _make_request("new", prompt_len=50)
        sync_scheduler.waiting.append(new_req)

        self._run_schedule(sync_scheduler)

        # new_req should be held back (running batch, no new batch)
        waiting_ids = [r.request_id for r in sync_scheduler.waiting]
        assert "new" in waiting_ids


class TestChunkedPrefillSpyreMixinSchedule:
    """ChunkedPrefillSpyreMixin bypasses constraints in async mode."""

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(ChunkedPrefillSpyreScheduler):
            s = ChunkedPrefillSpyreScheduler()
        _set_common_attrs(s)
        _set_chunked_prefill_attrs(s)
        return s

    @pytest.fixture
    def async_scheduler(self):
        with _patch_init(AsyncChunkedPrefillSpyreScheduler):
            s = AsyncChunkedPrefillSpyreScheduler()
        _set_common_attrs(s)
        _set_chunked_prefill_attrs(s)
        return s

    def test_async_applies_prefill_constraints(self, async_scheduler):
        """Only one new prefill is scheduled at a time.

        The pre-filter limits the waiting queue to 1 new request before
        delegating to the base scheduler.  The second request is held back in
        the holdback queue and restored to waiting after the schedule() call.
        """
        req1 = _make_request("req1", prompt_len=50)
        req2 = _make_request("req2", prompt_len=60)
        async_scheduler.waiting.append(req1)
        async_scheduler.waiting.append(req2)

        # Mock can_schedule_prefill to allow both requests
        with patch.object(async_scheduler, "can_schedule_prefill", return_value=True):
            # Only req1 reaches super().schedule() — the pre-filter limits to 1
            mock_output = _make_scheduler_output(["req1"], tokens_per_req=50)
            with patch(
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler.schedule",
                return_value=mock_output,
            ):
                result = async_scheduler.schedule()

        # Only req1 should be in ongoing_prefills
        assert len(async_scheduler.ongoing_prefills) == 1
        assert async_scheduler.ongoing_prefills[0].request_id == "req1"
        # req2 was never passed to super().schedule(); it is restored to waiting
        assert any(r.request_id == "req2" for r in async_scheduler.waiting)
        # Output contains req1 only
        assert "req1" in result.num_scheduled_tokens
        assert "req2" not in result.num_scheduled_tokens

    def test_sync_applies_prefill_constraints(self, sync_scheduler):
        """In sync mode, ongoing_prefills is updated and constraints are applied."""
        req = _make_request("req1", prompt_len=50)
        sync_scheduler.waiting.append(req)

        mock_output = Mock(spec=SchedulerOutput)
        mock_output.num_scheduled_tokens = {"req1": 50}
        mock_output.has_structured_output_requests = False

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.schedule",
            return_value=mock_output,
        ):
            sync_scheduler.schedule()

        # In sync mode the request should have been moved to ongoing_prefills
        assert any(r.request_id == "req1" for r in sync_scheduler.ongoing_prefills)

    def test_async_clears_speculatively_completed_prefill(self, async_scheduler):
        """With async scheduling, _update_after_schedule speculatively advances
        num_computed_tokens before update_from_output() is called.  When schedule()
        is called again (run-ahead via step_with_batch_queue), a request whose
        num_computed_tokens has already reached num_prompt_tokens must be:
        1. Removed from ongoing_prefills (its prefill is speculatively done), and
        2. Included in self.running so the base scheduler can schedule it for
           decode — this is safe because execution is strictly sequential (FIFO):
           execute(S_{k-1}) always completes before execute(S_k) starts, so
           _maybe_prepare_last_prefill in execute(S_{k-1}) has already added the
           request to input_batch by the time execute(S_k) calls _prepare_decode.
        """
        chunk_size = async_scheduler.chunk_size

        # req1 just completed its prefill: _update_after_schedule set
        # num_computed_tokens = num_prompt_tokens (speculative advance).
        req1 = _make_request("req1", prompt_len=chunk_size)
        req1.num_computed_tokens = chunk_size  # speculatively done

        # req2 and req3 are decoding concurrently.
        req2 = _make_request("req2", prompt_len=50)
        req2.num_computed_tokens = 50
        req3 = _make_request("req3", prompt_len=60)
        req3.num_computed_tokens = 60

        async_scheduler.ongoing_prefills = [req1]
        async_scheduler.running = [req1, req2, req3]
        # Simulate that req1's prefill was captured in a pending snapshot
        # (as would happen after the first schedule() call completes).
        async_scheduler._prefill_scheduled_num_computed_queue.append({"req1": 0})
        async_scheduler._scheduled_ongoing_prefill_ids = frozenset({"req1"})

        captured_running = []

        def _capture_and_return(*_args, **_kwargs):
            captured_running.extend(async_scheduler.running)
            return _make_scheduler_output(
                [r.request_id for r in async_scheduler.running], tokens_per_req=1
            )

        with patch(
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler.schedule",
            side_effect=_capture_and_return,
        ):
            async_scheduler.schedule()

        # req1's prefill is speculatively done — it must be visible to
        # super().schedule() so that it can be scheduled for its first decode.
        # FIFO execution guarantees that execute(S_{k-1}) which calls
        # _maybe_prepare_last_prefill(req1) always completes before
        # execute(S_k) which calls _prepare_decode, so req1 is always in
        # input_batch before the decode assertion fires.
        assert req1.request_id in [r.request_id for r in captured_running]
        # req2 and req3 are decoding and must reach super().schedule() normally.
        assert req2.request_id in [r.request_id for r in captured_running]
        assert req3.request_id in [r.request_id for r in captured_running]
        # ongoing_prefills should be empty after the cleanup
        assert async_scheduler.ongoing_prefills == []


class TestChunkedPrefillSpyreMixinUpdateFromOutput:
    """update_from_output() filters scheduler output in async mode."""

    @pytest.fixture
    def async_scheduler(self):
        with _patch_init(AsyncChunkedPrefillSpyreScheduler):
            s = AsyncChunkedPrefillSpyreScheduler()
        _set_common_attrs(s)
        _set_chunked_prefill_attrs(s)
        return s

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(ChunkedPrefillSpyreScheduler):
            s = ChunkedPrefillSpyreScheduler()
        _set_common_attrs(s)
        _set_chunked_prefill_attrs(s)
        return s

    def test_async_empty_model_output_creates_empty_scheduler_output(self, async_scheduler):
        """When model output is empty (scheduler ran ahead), async mode passes empty
        scheduler output to the base class."""
        sched_output = _make_scheduler_output(["req1", "req2"])
        model_output = _make_model_runner_output([])  # empty — no execution

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        # Base should receive an empty scheduler output
        assert captured["scheduler_output"].total_num_scheduled_tokens == 0
        assert captured["scheduler_output"].num_scheduled_tokens == {}

    def test_async_partial_execution_filters_scheduler_output(self, async_scheduler):
        """When only a subset of scheduled requests were executed, the scheduler output
        passed to the base class is filtered to only those executed."""
        sched_output = _make_scheduler_output(["req1", "req2", "req3"])
        # Only req1 and req3 were executed
        model_output = _make_model_runner_output(["req1", "req3"])

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        passed_output = captured["scheduler_output"]
        assert set(passed_output.num_scheduled_tokens.keys()) == {"req1", "req3"}
        assert "req2" not in passed_output.num_scheduled_tokens

    def test_async_all_executed_passes_original_scheduler_output(self, async_scheduler):
        """When all scheduled requests were executed, no filtering occurs."""
        sched_output = _make_scheduler_output(["req1", "req2"])
        model_output = _make_model_runner_output(["req1", "req2"])

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        # No filtering needed — same object should be passed through
        passed_output = captured["scheduler_output"]
        assert set(passed_output.num_scheduled_tokens.keys()) == {"req1", "req2"}

    def test_sync_empty_model_output_passes_through(self, sync_scheduler):
        """In sync mode, empty model output is passed through without creating a
        filtered copy (the async-only path is not triggered)."""
        sched_output = _make_scheduler_output([])
        model_output = _make_model_runner_output([])  # empty

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            sync_scheduler.update_from_output(sched_output, model_output)

        # In sync mode the original (empty) scheduler output is passed through
        assert captured["scheduler_output"] is sched_output

    def test_async_tkv_updated_from_model_output(self, async_scheduler):
        """tkv is updated from model_runner_output even in async mode."""
        sched_output = _make_scheduler_output(["req1"])
        model_output = _make_model_runner_output(["req1"], tkv=512)

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        assert async_scheduler.tkv == 512

    def test_async_awaiting_commit_stays_true_while_snapshots_pending(self, async_scheduler):
        """_schedule_awaiting_commit must remain True until all pending snapshot
        entries are consumed.

        With batch_queue_size=2, a second schedule() runs (and appends a snapshot)
        *before* update_from_output() for the first batch is called.  The queue
        therefore has TWO entries when update_from_output() first fires, and the
        flag must stay True so that the run-ahead guard fires again for the third
        schedule() call — preventing the scheduler from re-scheduling the same
        prefill chunk that is still executing.

        Regression test for the multi-chunk prefill deadlock with TP=2.
        """
        chunk_size = async_scheduler.chunk_size  # 128
        prompt_len = chunk_size + 22  # 150 tokens → needs 2 chunks

        req1 = _make_request("req1", prompt_len=prompt_len)
        async_scheduler.ongoing_prefills = [req1]
        async_scheduler.running = [req1]

        # Simulate two schedule() calls having already happened:
        #   - schedule1 appended {req1: 0}   (chunk1 was scheduled)
        #   - schedule2 appended {req1: 128} (chunk2 was scheduled as run-ahead)
        # Both are still pending (neither has been committed yet).
        async_scheduler._prefill_scheduled_num_computed_queue.extend(
            [{req1.request_id: 0}, {req1.request_id: chunk_size}]
        )
        async_scheduler._scheduled_ongoing_prefill_ids = frozenset({req1.request_id})
        async_scheduler._schedule_awaiting_commit = True

        # update_from_output for the first batch (chunk1):
        # req1.num_computed_tokens is updated to chunk_size (committed state).
        req1.num_computed_tokens = chunk_size

        sched_output = _make_scheduler_output([req1.request_id], tokens_per_req=chunk_size)
        model_output = _make_model_runner_output([req1.request_id], tkv=chunk_size)

        with patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output"):
            async_scheduler.update_from_output(sched_output, model_output)

        # After committing chunk1, one snapshot ({req1:128}) still remains.
        # The flag must stay True so the guard fires for schedule3.
        assert async_scheduler._schedule_awaiting_commit is True, (
            "_schedule_awaiting_commit must remain True while snapshot queue is non-empty "
            "to prevent re-scheduling in-flight prefill chunks (multi-chunk deadlock)"
        )
        assert len(async_scheduler._prefill_scheduled_num_computed_queue) == 1

        # Simulate commit2 (chunk2): queue is now drained to zero.
        req1.num_computed_tokens = prompt_len  # chunk2 committed
        sched_output2 = _make_scheduler_output([req1.request_id], tokens_per_req=22)
        model_output2 = _make_model_runner_output([req1.request_id], tkv=prompt_len)

        with patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output"):
            async_scheduler.update_from_output(sched_output2, model_output2)

        # Now the queue is empty → flag may be cleared.
        assert async_scheduler._schedule_awaiting_commit is False, (
            "_schedule_awaiting_commit should be False once all snapshot entries consumed"
        )
        assert len(async_scheduler._prefill_scheduled_num_computed_queue) == 0
