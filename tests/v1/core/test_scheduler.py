"""Unit tests for Spyre scheduler classes (sync and async variants).

Key behaviours under test:
  - PoolingSpyreScheduler.schedule() applies warmup-shape constraints
  - ChunkedPrefillSpyreScheduler.schedule() applies the Spyre chunked-prefill
    constraints in both sync and async modes (one prefill at a time, no mixing
    prefill and decode); ongoing_prefills is derived from self.running on
    every schedule() call (no snapshot state)
  - ChunkedPrefillSpyreScheduler._update_after_schedule() bumps a per-request
    counter (``_inflight_prefill_tokens``) by the optimistic prefill-chunk
    advance so update_from_output() can later reconcile the actual committed
    position
  - ChunkedPrefillSpyreScheduler.update_from_output() reconciles the
    optimistic advance using left-padding / prefix-cache info from the runner
    and decrements ``_inflight_prefill_tokens``; under async mode it also
    filters scheduler output to only include requests that were actually
    executed in this step
"""

import pytest
from unittest.mock import Mock, patch

from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request

from sendnn_inference.v1.core.async_scheduler import (
    AsyncChunkedPrefillSpyreScheduler,
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


def _make_model_runner_output(
    req_ids: list[str],
    tkv: int = 10,
    left_padding: dict[str, int] | None = None,
    prefix_cache_hit_len: dict[str, int] | None = None,
) -> SpyreModelRunnerOutput:
    return SpyreModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[1]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        left_padding=left_padding or {},
        prefix_cache_hit_len=prefix_cache_hit_len or {},
        tkv=tkv,
    )


def _patch_init(cls):
    """Context manager: bypass __init__ for a scheduler class."""
    return patch.object(cls, "__init__", lambda self, *a, **kw: None)


def _set_common_attrs(scheduler, chunk_size=128, max_seqs=4, async_scheduling=False):
    """Set the minimum set of attributes needed by the scheduler methods."""
    mock_vllm_config = Mock()
    mock_vllm_config.model_config.max_model_len = 2048
    mock_vllm_config.scheduler_config.max_num_batched_tokens = chunk_size
    mock_vllm_config.scheduler_config.max_num_seqs = max_seqs
    mock_vllm_config.scheduler_config.async_scheduling = async_scheduling

    scheduler.vllm_config = mock_vllm_config
    scheduler.model_config = mock_vllm_config.model_config
    scheduler.scheduler_config = mock_vllm_config.scheduler_config
    scheduler.waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler.max_num_running_reqs = max_seqs
    # ``self.requests`` is used by _update_after_schedule and the placeholder
    # reconciliation in update_from_output.
    scheduler.requests = {}


def _set_chunked_prefill_attrs(scheduler, chunk_size=128):
    """Set extra attributes required by ChunkedPrefillSpyreScheduler."""
    scheduler.chunk_size = chunk_size
    scheduler.ongoing_prefills = []
    scheduler._inflight_prefill_tokens = {}
    scheduler.do_interleaving = False
    scheduler.previous_step_was_prefill = False
    scheduler.tkv = 0
    scheduler.block_size = 64
    scheduler.max_batch_tkv_limit = 131072
    # skipped_waiting holds grammar-blocked requests in the base vLLM Scheduler
    scheduler.skipped_waiting = FCFSRequestQueue()


def _set_pooling_attrs(scheduler):
    """Set extra attributes required by PoolingSpyreScheduler."""
    scheduler.spyre_warmup_shapes = (
        {"prompt_length": 64, "batch_size": 4},
        {"prompt_length": 128, "batch_size": 2},
    )


class TestPoolingSpyreSchedulerSchedule:
    """PoolingSpyreScheduler applies warmup-shape constraints."""

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(PoolingSpyreScheduler):
            s = PoolingSpyreScheduler()
        _set_common_attrs(s, async_scheduling=False)
        _set_pooling_attrs(s)
        return s

    def _run_schedule(self, scheduler):
        """Patch the base scheduler.schedule() and call our subclass."""
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
        req_too_long = _make_request("too_long", prompt_len=200)
        req_fits = _make_request("fits", prompt_len=50)
        sync_scheduler.waiting.append(req_too_long)
        sync_scheduler.waiting.append(req_fits)

        self._run_schedule(sync_scheduler)

        waiting_ids = [r.request_id for r in sync_scheduler.waiting]
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
        """When requests are already running, the waiting queue is held back."""
        running_req = _make_request("running", prompt_len=50)
        sync_scheduler.running = [running_req]
        new_req = _make_request("new", prompt_len=50)
        sync_scheduler.waiting.append(new_req)

        self._run_schedule(sync_scheduler)

        waiting_ids = [r.request_id for r in sync_scheduler.waiting]
        assert "new" in waiting_ids


class TestChunkedPrefillSpyreSchedulerSchedule:
    """ChunkedPrefillSpyreScheduler applies prefill constraints in both modes."""

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(ChunkedPrefillSpyreScheduler):
            s = ChunkedPrefillSpyreScheduler()
        _set_common_attrs(s, async_scheduling=False)
        _set_chunked_prefill_attrs(s)
        return s

    @pytest.fixture
    def async_scheduler(self):
        with _patch_init(AsyncChunkedPrefillSpyreScheduler):
            s = AsyncChunkedPrefillSpyreScheduler()
        _set_common_attrs(s, async_scheduling=True)
        _set_chunked_prefill_attrs(s)
        return s

    def test_async_applies_prefill_constraints(self, async_scheduler):
        """Only one new prefill is scheduled at a time."""
        req1 = _make_request("req1", prompt_len=50)
        req2 = _make_request("req2", prompt_len=60)
        async_scheduler.waiting.append(req1)
        async_scheduler.waiting.append(req2)

        with patch.object(async_scheduler, "can_schedule_prefill", return_value=True):
            mock_output = _make_scheduler_output(["req1"], tokens_per_req=50)
            with patch(
                "vllm.v1.core.sched.async_scheduler.AsyncScheduler.schedule",
                return_value=mock_output,
            ):
                result = async_scheduler.schedule()

        assert len(async_scheduler.ongoing_prefills) == 1
        assert async_scheduler.ongoing_prefills[0].request_id == "req1"
        assert any(r.request_id == "req2" for r in async_scheduler.waiting)
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

        assert any(r.request_id == "req1" for r in sync_scheduler.ongoing_prefills)

    def test_ongoing_prefills_derived_from_running_inflight(self, async_scheduler):
        """ongoing_prefills is recomputed from self.running on each schedule()
        based on the committed position. A request whose
        ``num_computed_tokens`` was optimistically advanced past prompt_len
        but whose advance is still pending (``_inflight_prefill_tokens > 0``)
        is still prefilling and SHOULD appear in ongoing_prefills."""
        chunk_size = async_scheduler.chunk_size
        prompt_len = chunk_size + 22  # 150 tokens

        # req_done: fully committed, no inflight tokens → not prefilling.
        req_done = _make_request("req_done", prompt_len=prompt_len)
        req_done.num_computed_tokens = prompt_len

        # req_inflight: optimistically advanced full prefill, but the advance
        # is still in flight (counter == prompt_len), so the committed
        # position is 0 → still prefilling.
        req_inflight = _make_request("req_inflight", prompt_len=prompt_len)
        req_inflight.num_computed_tokens = prompt_len
        async_scheduler._inflight_prefill_tokens["req_inflight"] = prompt_len

        async_scheduler.running = [req_done, req_inflight]

        with patch(
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler.schedule",
            return_value=_make_scheduler_output([], tokens_per_req=0),
        ):
            async_scheduler.schedule()

        ongoing_ids = {r.request_id for r in async_scheduler.ongoing_prefills}
        assert "req_done" not in ongoing_ids
        assert "req_inflight" in ongoing_ids


class TestUpdateAfterScheduleBumpsPlaceholders:
    """_update_after_schedule bumps _inflight_prefill_tokens for prefill chunks."""

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(ChunkedPrefillSpyreScheduler):
            s = ChunkedPrefillSpyreScheduler()
        _set_common_attrs(s, async_scheduling=False)
        _set_chunked_prefill_attrs(s)
        return s

    def test_prefill_chunk_bumps_placeholder(self, sync_scheduler):
        """A prefill-chunk advance is recorded in _inflight_prefill_tokens."""
        chunk_size = sync_scheduler.chunk_size
        prompt_len = chunk_size + 22

        req = _make_request("req1", prompt_len=prompt_len)
        sync_scheduler.requests["req1"] = req
        sched_output = _make_scheduler_output(["req1"], tokens_per_req=chunk_size)

        # Stand-in for the base scheduler's optimistic advance.
        def fake_base_update_after(self_inner, sched_out):
            for rid, n in sched_out.num_scheduled_tokens.items():
                r = self_inner.requests[rid]
                r.num_computed_tokens += n
                r.is_prefill_chunk = r.num_computed_tokens < (
                    r.num_tokens + r.num_output_placeholders
                )

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule",
            new=fake_base_update_after,
        ):
            sync_scheduler._update_after_schedule(sched_output)

        # First chunk of a multi-chunk prefill: counter bumped to chunk_size.
        assert req.num_computed_tokens == chunk_size
        assert sync_scheduler._inflight_prefill_tokens["req1"] == chunk_size
        # Crucially, num_output_placeholders is NOT touched (we use a private
        # dict to avoid breaking upstream's num_new_tokens formula).
        assert req.num_output_placeholders == 0

    def test_last_prefill_chunk_does_not_bump_placeholder(self, sync_scheduler):
        """The final chunk that completes the prefill does not bump the counter.

        The base scheduler clears ``is_prefill_chunk`` for the last chunk
        (because num_computed_tokens reaches num_tokens), so the override
        should not record an in-flight count for it.
        """
        chunk_size = sync_scheduler.chunk_size
        prompt_len = chunk_size  # exactly one chunk

        req = _make_request("req1", prompt_len=prompt_len)
        sync_scheduler.requests["req1"] = req
        sched_output = _make_scheduler_output(["req1"], tokens_per_req=chunk_size)

        def fake_base_update_after(self_inner, sched_out):
            for rid, n in sched_out.num_scheduled_tokens.items():
                r = self_inner.requests[rid]
                r.num_computed_tokens += n
                r.is_prefill_chunk = r.num_computed_tokens < (
                    r.num_tokens + r.num_output_placeholders
                )

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule",
            new=fake_base_update_after,
        ):
            sync_scheduler._update_after_schedule(sched_output)

        assert req.num_computed_tokens == chunk_size
        assert "req1" not in sync_scheduler._inflight_prefill_tokens


class TestChunkedPrefillSpyreSchedulerUpdateFromOutput:
    """update_from_output() reconciles placeholders and filters in async mode."""

    @pytest.fixture
    def async_scheduler(self):
        with _patch_init(AsyncChunkedPrefillSpyreScheduler):
            s = AsyncChunkedPrefillSpyreScheduler()
        _set_common_attrs(s, async_scheduling=True)
        _set_chunked_prefill_attrs(s)
        return s

    @pytest.fixture
    def sync_scheduler(self):
        with _patch_init(ChunkedPrefillSpyreScheduler):
            s = ChunkedPrefillSpyreScheduler()
        _set_common_attrs(s, async_scheduling=False)
        _set_chunked_prefill_attrs(s)
        return s

    def test_async_empty_model_output_creates_empty_scheduler_output(self, async_scheduler):
        """When model output is empty (scheduler ran ahead), pass an empty
        scheduler output to the base class so the in-flight scheduling is not
        prematurely committed."""
        sched_output = _make_scheduler_output(["req1", "req2"])
        model_output = _make_model_runner_output([])

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        assert captured["scheduler_output"].total_num_scheduled_tokens == 0
        assert captured["scheduler_output"].num_scheduled_tokens == {}

    def test_async_partial_execution_filters_scheduler_output(self, async_scheduler):
        """When only a subset of scheduled requests were executed, the scheduler output
        passed to the base class is filtered to only those executed."""
        sched_output = _make_scheduler_output(["req1", "req2", "req3"])
        model_output = _make_model_runner_output(["req1", "req3"])

        # Register requests so the placeholder reconciliation has someone to
        # look up (no-op since placeholders are 0).
        for rid in ("req1", "req2", "req3"):
            async_scheduler.requests[rid] = _make_request(rid, prompt_len=200)

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
        sched_output = _make_scheduler_output(["req1", "req2"])
        model_output = _make_model_runner_output(["req1", "req2"])

        for rid in ("req1", "req2"):
            async_scheduler.requests[rid] = _make_request(rid, prompt_len=200)

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        passed_output = captured["scheduler_output"]
        assert set(passed_output.num_scheduled_tokens.keys()) == {"req1", "req2"}

    def test_sync_empty_model_output_passes_through(self, sync_scheduler):
        sched_output = _make_scheduler_output([])
        model_output = _make_model_runner_output([])

        captured = {}

        def fake_super_update(scheduler_output, model_runner_output):
            captured["scheduler_output"] = scheduler_output

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
            side_effect=fake_super_update,
        ):
            sync_scheduler.update_from_output(sched_output, model_output)

        assert captured["scheduler_output"] is sched_output

    def test_async_tkv_updated_from_model_output(self, async_scheduler):
        sched_output = _make_scheduler_output(["req1"])
        model_output = _make_model_runner_output(["req1"], tkv=512)
        async_scheduler.requests["req1"] = _make_request("req1", prompt_len=200)

        with patch(
            "vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
        ):
            async_scheduler.update_from_output(sched_output, model_output)

        assert async_scheduler.tkv == 512

    def test_placeholder_cleared_on_commit(self, async_scheduler):
        """After update_from_output runs for an executed prefill chunk, the
        in-flight counter bump made in _update_after_schedule is cleared."""
        chunk_size = async_scheduler.chunk_size
        prompt_len = chunk_size + 22

        req = _make_request("req1", prompt_len=prompt_len)
        # Simulate the optimistic advance + counter bump that
        # _update_after_schedule would have done.
        req.num_computed_tokens = chunk_size
        async_scheduler._inflight_prefill_tokens["req1"] = chunk_size
        async_scheduler.requests["req1"] = req

        sched_output = _make_scheduler_output(["req1"], tokens_per_req=chunk_size)
        model_output = _make_model_runner_output(
            ["req1"],
            tkv=chunk_size,
            left_padding={"req1": 0},
            prefix_cache_hit_len={"req1": 0},
        )

        with patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output"):
            async_scheduler.update_from_output(sched_output, model_output)

        # No left-padding / prefix-cache, so num_computed_tokens stays at
        # chunk_size; counter cleared.
        assert req.num_computed_tokens == chunk_size
        assert "req1" not in async_scheduler._inflight_prefill_tokens

    def test_first_chunk_left_padding_correction(self, async_scheduler):
        """When the first prefill chunk has left padding, num_computed_tokens
        is corrected downward by the padding amount and the counter is
        cleared."""
        chunk_size = async_scheduler.chunk_size
        prompt_len = chunk_size * 2  # two chunks
        left_padding = 10

        req = _make_request("req1", prompt_len=prompt_len)
        req.num_computed_tokens = chunk_size  # optimistic
        async_scheduler._inflight_prefill_tokens["req1"] = chunk_size
        async_scheduler.requests["req1"] = req

        sched_output = _make_scheduler_output(["req1"], tokens_per_req=chunk_size)
        model_output = _make_model_runner_output(
            ["req1"],
            tkv=chunk_size,
            left_padding={"req1": left_padding},
            prefix_cache_hit_len={"req1": 0},
        )

        with patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output"):
            async_scheduler.update_from_output(sched_output, model_output)

        # adjust_computed_tokens(chunk_size, left_padding=10, prefix=0)
        # returns chunk_size - 10. Correction: -10.
        assert req.num_computed_tokens == chunk_size - left_padding
        assert "req1" not in async_scheduler._inflight_prefill_tokens

    def test_first_chunk_correction_under_run_ahead(self, async_scheduler):
        """Regression test: when chunk 1 commits while chunk 2 has already
        been speculatively scheduled (async run-ahead), the per-chunk
        ``is_last_chunk`` decision must be based on ``committed_before +
        scheduled_n_for_this_chunk``, NOT on the cumulative
        ``req.num_computed_tokens`` (which by then includes chunk 2's
        run-ahead advance and would otherwise look like the prefill is
        already done, causing the left-padding correction to be skipped).
        """
        chunk_size = async_scheduler.chunk_size
        chunk_1_n = chunk_size
        chunk_2_n = 22
        prompt_len = chunk_1_n + chunk_2_n
        left_padding = 10

        req = _make_request("req1", prompt_len=prompt_len)
        # Both chunks have been scheduled and the base scheduler has
        # optimistically advanced num_computed_tokens past prompt_len.
        # The cumulative inflight counter records both chunks.
        req.num_computed_tokens = prompt_len
        async_scheduler._inflight_prefill_tokens["req1"] = chunk_1_n + chunk_2_n
        async_scheduler.requests["req1"] = req

        # Now chunk 1 commits.
        sched_output = _make_scheduler_output(["req1"], tokens_per_req=chunk_1_n)
        model_output = _make_model_runner_output(
            ["req1"],
            tkv=chunk_1_n,
            left_padding={"req1": left_padding},
            prefix_cache_hit_len={"req1": 0},
        )

        with patch("vllm.v1.core.sched.scheduler.Scheduler.update_from_output"):
            async_scheduler.update_from_output(sched_output, model_output)

        # The correction MUST fire (this is the run-ahead case the snapshot
        # mechanism originally handled). num_computed_tokens shifts down by
        # left_padding; chunk 2's advance is preserved.
        assert req.num_computed_tokens == prompt_len - left_padding
        # Inflight is decremented by chunk_1_n; chunk 2 is still in flight.
        assert async_scheduler._inflight_prefill_tokens["req1"] == chunk_2_n
