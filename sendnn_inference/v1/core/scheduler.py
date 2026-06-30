# SPDX-License-Identifier: Apache-2.0

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Union


from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request, RequestStatus

import sendnn_inference.envs as envs_spyre
from sendnn_inference.platform import SpyrePlatform
from sendnn_inference.v1.worker.spyre_model_runner import SpyreModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    SchedulerOutput = None

logger = init_logger(__name__)


@dataclass
class SpyreBenchState:
    """Bench-metrics-only state for tracking per-request and per-step timing.
    Only instantiated when SENDNN_INFERENCE_BENCH_METRICS_ENABLED is set."""

    chunk_latencies: dict[str, list[float]] = field(default_factory=dict)
    arrival_ts: dict[str, float] = field(default_factory=dict)
    first_scheduled_ts: dict[str, float] = field(default_factory=dict)
    chunk_start_times: dict[str, list[float]] = field(default_factory=dict)
    decode_latencies: dict[str, list[float]] = field(default_factory=dict)
    decode_start_times: dict[str, list[float]] = field(default_factory=dict)
    tkvs: dict[str, list[int]] = field(default_factory=dict)
    left_padding_blocks: dict[str, list[int]] = field(default_factory=dict)
    prefill_step_start: float | None = None
    decode_step_start: float | None = None


# Ensure that block_size is 64
# This ensures the rounding function is correct
assert SpyrePlatform.get_block_size() == 64


def round_up_to_block_size(n: int) -> int:
    # Helper function to round up to the nearest block size
    # Uses bitwise alignment for better performance
    return (n + 63) & ~63


class SpyreScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support static
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config


class PoolingSpyreScheduler(SpyreScheduler):
    """Support of pooling models"""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize SpyreScheduler
        super().__init__(*args, **kwargs)

        # Add our own state for handling Spyre constraints:
        # all warmup shapes that we can support
        self.spyre_warmup_shapes: tuple[dict[str, int], ...] = SpyrePlatform.get_warmup_shapes(
            self.scheduler_config
        )

    def schedule(self) -> SchedulerOutput:
        """This override adds constraints and then delegates most of the work
        to the base scheduler"""
        # First purge the full waiting queue into our holdback queue, preserving
        # priority, so that the base scheduler does not see them.
        # This lets us ensure that the set of requests scheduled have at least
        # one common warmup shape.
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.popleft())

        # store requests which don't fit the warmup shapes of the current batch
        skip_queue: deque[Request] = deque()

        # If no requests are currently running, we can now release requests back
        # into the waiting queue in priority order for the scheduler to prefill.
        # These must share a common warmup shape
        if len(self.running) == 0:
            # Make a copy of the warmup shapes
            available_warmup_shapes = list(self.spyre_warmup_shapes)
            last_available_warmup_shapes = available_warmup_shapes

            while holdback_queue:
                request = holdback_queue[0]

                # prune the possible shapes to only those that fit this request
                # and the growing batch size
                available_warmup_shapes = self._get_matching_warmup_shapes(
                    request=request,
                    warmup_shapes=available_warmup_shapes,
                    current_batch_size=len(self.waiting),
                )

                if len(available_warmup_shapes) > 0:
                    # There is still at least one valid shape, so add to the
                    # waiting queue
                    self.waiting.append(holdback_queue.popleft())
                    # remember the available warmup shapes of the current batch
                    last_available_warmup_shapes = available_warmup_shapes
                else:
                    # calculating the max possible batch size among the
                    # available warmup shapes of the scheduled requests
                    max_batch = max([d["batch_size"] for d in last_available_warmup_shapes])

                    # if there is potential space in the batch but the current
                    # request does not fit, skip it and try with the next
                    if len(self.waiting) < max_batch:
                        available_warmup_shapes = last_available_warmup_shapes
                        skip_queue.append(holdback_queue.popleft())
                    else:
                        # If the batch is full, we exit the loop here
                        break

            logger.debug(
                "Scheduling a new batch of %d requests, holding back %d requests",
                len(self.waiting),
                len(holdback_queue),
            )
        else:
            logger.debug("Scheduling a running batch of %d requests", len(self.running))

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # first move skipped and then unscheduled requests back
        # to the waiting queue, preserving priority
        while skip_queue:
            self.waiting.append(skip_queue.popleft())

        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        outputs._spyre_grammar_output = self.get_grammar_bitmask(outputs)  # type: ignore[attr-defined]
        return outputs

    def _get_matching_warmup_shapes(
        self, request: Request, warmup_shapes: list[dict[str, int]], current_batch_size: int
    ) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        return [
            shape
            for shape in warmup_shapes
            if request.num_prompt_tokens <= shape["prompt_length"]
            and current_batch_size < shape["batch_size"]
        ]


class ChunkedPrefillSpyreScheduler(SpyreScheduler):
    """
    Chunked-Prefill Scheduling policy

    The prefill vs. decode priority policy is the following:
        - Current prefill request priority: A new request cannot start prefill
           while another request's prefill is on-going

        - Prefill step interleaving: The prefill steps are interleaved with
            one decode step: as long as there are decoding requests, two
            prefill steps cannot be consecutive

        - General prefill priority: conditioned on interleaving constraint,
            prefill has priority over decode

        - No empty step: if a prefill step is prevented because it doesn't
            satisfy Spyre's specific constraints, a decode step is scheduled

    Spyre scheduling constraints:

        - Prefill batch size: prefill batch is of size 1, only one request's
            chunked prefill can be scheduled at a time

        - Decode batch size: cannot have more than max_num_seqs running
            requests, including prefill and decode

        Note: all the remaining constraints need to be satisfied at the time
            of scheduling the last chunk of a chunked prefill

        - Max model length constraint: the number of requested tokens must fit
            between the maximum TKV of all the running requests and the end of
            the model's context

        - Volumetric constraint: the total "surface" defined by the running
            requests should never exceed `VLLM_DT_MAX_BATCH_TKV_LIMIT`. See
            `check_batch_tkv_limit()` method for details.

        - The surface defined by the maximum TKV of
            all the running requests and the number of running requests must
            not exceed the limit defined by `VLLM_DT_MAX_BATCH_TKV_LIMIT`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.chunk_size = self.scheduler_config.max_num_batched_tokens

        # We want to keep track of requests for which the prefill is ongoing.
        # Theoretically, only one request can be prefilled at a time, but we
        # keep a list to be able to batch prefills in the future.
        self.ongoing_prefills: list[Request] = []

        # Prefills interleaving: if the feature flag is set, prefill operations
        # are interleaved with a decode step. This allows to minimize currently
        # decoding requests
        self.do_interleaving: bool = envs_spyre.SENDNN_INFERENCE_CP_INTERLEAVE_STEPS
        self.previous_step_was_prefill: bool = False

        self.tkv = 0
        self.block_size = SpyrePlatform.get_block_size()
        self.max_batch_tkv_limit = SpyrePlatform.get_max_batch_tkv_limit()

        self._bench: SpyreBenchState | None = None

        if envs_spyre.SENDNN_INFERENCE_BENCH_METRICS_ENABLED:
            self._bench = SpyreBenchState()

        assert self.max_batch_tkv_limit != -1, (
            "Expecting the env var VLLM_DT_MAX_BATCH_TKV_LIMIT to be set in platform.py"
        )

        self.total_reserved_blocks = 0
        self.reserved_blocks = dict[str, int]()

    def update_from_output(self, scheduler_output, model_runner_output):
        assert isinstance(model_runner_output, SpyreModelRunnerOutput), (
            "Expecting an instance of CPSpyreModelRunnerOutput when doing chunked prefill."
        )

        # Measure timing durations and accumulate metrics (scheduler-side timing injection)
        if self._bench is not None:
            self._bench_update_from_output(scheduler_output, model_runner_output)

        # Remove completed prefills
        self.ongoing_prefills = [
            req for req in self.ongoing_prefills if req.num_computed_tokens < req.num_prompt_tokens
        ]

        self.tkv = model_runner_output.tkv
        result = super(SpyreScheduler, self).update_from_output(
            scheduler_output, model_runner_output
        )

        for finished_request in self.finished_req_ids:
            blocks = self.reserved_blocks.pop(finished_request, 0)
            self.total_reserved_blocks -= blocks
            assert self.total_reserved_blocks >= 0

        return result

    def _bench_update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: SpyreModelRunnerOutput,
    ) -> None:
        assert self._bench is not None
        now = time.time()

        if self._bench.prefill_step_start is not None:
            assert self.previous_step_was_prefill and self._bench.decode_step_start is None
            t0 = self._bench.prefill_step_start
            duration = now - t0
            all_prefill_reqs = [
                r.req_id for r in scheduler_output.scheduled_new_reqs
            ] + scheduler_output.scheduled_cached_reqs.req_ids
            for req_id in all_prefill_reqs:
                self._bench.chunk_latencies.setdefault(req_id, []).append(duration)
                self._bench.chunk_start_times.setdefault(req_id, []).append(t0)
                self._bench.tkvs.setdefault(req_id, []).append(model_runner_output.tkv)
            self._bench.prefill_step_start = None
            self._bench.decode_step_start = None

        elif self._bench.decode_step_start is not None:
            assert not self.previous_step_was_prefill and self._bench.prefill_step_start is None
            t0 = self._bench.decode_step_start
            duration = now - t0
            tkv = model_runner_output.tkv
            max_num_blocks = math.ceil(tkv / self.block_size)
            req_by_id = {r.request_id: r for r in self.running}
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self._bench.decode_latencies.setdefault(req_id, []).append(duration)
                self._bench.decode_start_times.setdefault(req_id, []).append(t0)
                self._bench.tkvs.setdefault(req_id, []).append(tkv)
                req = req_by_id.get(req_id)
                if req is not None:
                    req_num_blocks = math.ceil(req.num_computed_tokens / self.block_size)
                    self._bench.left_padding_blocks.setdefault(req_id, []).append(
                        max_num_blocks - req_num_blocks
                    )
            self._bench.prefill_step_start = None
            self._bench.decode_step_start = None

        for req in self.ongoing_prefills:
            if req.request_id not in self._bench.first_scheduled_ts:
                self._bench.first_scheduled_ts[req.request_id] = now
                self._bench.arrival_ts[req.request_id] = req.arrival_time

    def get_and_clear_chunk_stats(self, req_id: str) -> dict | None:
        """Return and clear accumulated chunk timing for a finished request."""
        if self._bench is None:
            return None
        lats = self._bench.chunk_latencies.pop(req_id, None)
        starts = self._bench.chunk_start_times.pop(req_id, None)
        dec_lats = self._bench.decode_latencies.pop(req_id, None)
        dec_starts = self._bench.decode_start_times.pop(req_id, None)
        tkvs = self._bench.tkvs.pop(req_id, None)
        left_padding_blocks = self._bench.left_padding_blocks.pop(req_id, None)
        if lats is None and dec_lats is None:
            return None
        return {
            "num_chunked_prefills": len(lats) if lats else 0,
            "chunk_prefill_latencies_s": lats or [],
            "chunk_prefill_start_times_s": starts or [],
            "decode_latencies_s": dec_lats or [],
            "decode_start_times_s": dec_starts or [],
            "tkvs": tkvs or [],
            "left_padding_blocks": left_padding_blocks or [],
        }

    def _free_request(self, request, delay_free_blocks: bool = False):
        """Override to inject Spyre bench metrics into kv_transfer_params so
        they travel over ZMQ to the API server process in EngineCoreOutput."""
        kv_xfer_params = super()._free_request(request, delay_free_blocks)

        if self._bench is not None:
            req_id = request.request_id
            chunk_stats = self.get_and_clear_chunk_stats(req_id)
            first_ts = self._bench.first_scheduled_ts.pop(req_id, None)
            arrival_ts = self._bench.arrival_ts.pop(req_id, None)
            # NOTE: queued_time_s looks like a duplicate of FinishedRequestStats.queued_time, but
            # it is not. FinishedRequestStats.queued_time is (scheduled_ts - queued_ts): both
            # timestamps are recorded inside the engine core, so it misses the time the request
            # spent in transit from the API server to the engine (IPC hop). Here we use the stamp
            # of the API server on receipt (request.arrival_time), which gives a complete client-
            # visible queue wait that adds up cleanly with prefill and decode latencies when
            # reconstructing TTFT.
            queued_time_s = (
                (first_ts - arrival_ts) if first_ts is not None and arrival_ts is not None else 0.0
            )
            num_executed = chunk_stats["num_chunked_prefills"] if chunk_stats else 0
            num_expected = math.ceil(request.num_prompt_tokens / self.chunk_size)
            num_skipped = max(0, num_expected - num_executed)
            cache_hit_pct = num_skipped / num_expected if num_expected > 0 else 0.0
            spyre_data = {
                "queued_time_s": queued_time_s,
                "num_chunked_prefills": num_executed,
                "chunk_prefill_latencies_s": chunk_stats["chunk_prefill_latencies_s"]
                if chunk_stats
                else [],
                "chunk_prefill_start_times_s": chunk_stats["chunk_prefill_start_times_s"]
                if chunk_stats
                else [],
                "decode_latencies_s": chunk_stats["decode_latencies_s"] if chunk_stats else [],
                "decode_start_times_s": chunk_stats["decode_start_times_s"] if chunk_stats else [],
                "tkvs": chunk_stats["tkvs"] if chunk_stats else [],
                "prefix_cache_hit_pct": cache_hit_pct,
                "left_padding_blocks": chunk_stats["left_padding_blocks"] if chunk_stats else [],
            }
            if kv_xfer_params is None:
                kv_xfer_params = {"__spyre__": spyre_data}
            else:
                kv_xfer_params["__spyre__"] = spyre_data

        return kv_xfer_params

    def _current_chunk_token_threshold(self, new_prefill_candidates: list[Request]) -> int:
        """Returns the `long_prefill_token_threshold` to use for this step.

        For the chunk-0 step cap to `chunk_size - left_padding` so the base
        scheduler is aware of the padding blocks.
        Otherwise return `chunk_size`: the natural chunk boundary."""

        # If there are no new prefill candidates, no cap is needed.
        if not new_prefill_candidates:
            return self.chunk_size

        new_prefill = new_prefill_candidates[0]

        # Calculate left-padding tokens for this prompt.
        prompt_len = new_prefill.num_prompt_tokens
        n_chunks = math.ceil(prompt_len / self.chunk_size)
        padded_prompt_len = math.ceil(prompt_len / self.block_size) * self.block_size
        left_padding = n_chunks * self.chunk_size - padded_prompt_len

        # If the prefix cache already covers chunk 0's real content, no cap is
        # needed: the base scheduler will start from chunk i>=1, which has no
        # padding. `get_computed_blocks` records into `prefix_cache_stats` as
        # a side effect; the base scheduler calls it again, so toggle
        # log_stats off here to avoid double-counting.
        prev_log_stats = self.kv_cache_manager.log_stats
        self.kv_cache_manager.log_stats = False
        _, prefix_token_len = self.kv_cache_manager.get_computed_blocks(new_prefill)
        self.kv_cache_manager.log_stats = prev_log_stats
        if prefix_token_len >= self.chunk_size - left_padding:
            return self.chunk_size

        # Adjust the token threshold to account for left padding
        return self.chunk_size - left_padding

    def _get_required_blocks(self, request: Request, max_output: bool = False) -> tuple[int, int]:
        """
        Returns the block parameters for the given request.
        """
        # This basically replicates what the scheduler already does, but
        # scattered all over the place in `schedule()`
        if request.num_computed_tokens == 0:
            old_log_stats = self.kv_cache_manager.log_stats
            self.kv_cache_manager.log_stats = False
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            self.kv_cache_manager.log_stats = old_log_stats
            num_computed_tokens = num_new_local_computed_tokens
        else:
            new_computed_blocks = self.kv_cache_manager.create_kv_cache_blocks(blocks=tuple())
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        num_tokens = request.num_tokens
        if max_output:
            assert request.sampling_params is not None
            assert request.sampling_params.max_tokens is not None
            prompt_tokens = request.num_prompt_tokens
            max_tokens = request.sampling_params.max_tokens
            num_tokens = prompt_tokens + max_tokens

        num_blocks_to_allocate = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens,
            new_computed_blocks=new_computed_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=num_computed_tokens,
            num_tokens_main_model=num_tokens,
        )

        cached_blocks = sum(1 for block in new_computed_blocks.blocks[0] if block.ref_cnt > 0)
        total_blocks = math.ceil(num_tokens / self.block_size)
        assert cached_blocks + num_blocks_to_allocate == total_blocks
        return cached_blocks, num_blocks_to_allocate

    def _get_free_blocks(self) -> int:
        return self.kv_cache_manager.block_pool.get_num_free_blocks()

    def schedule(self) -> "SchedulerOutput":
        """
        The chunked prefill scheduling policy is enforced in this method, then
        delegates the final scheduling decision to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after
        """
        # First purge the full waiting queue into our holdback queue, preserving
        # priority, so that the base scheduler does not see them.
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.popleft())
        # Also drain skipped_waiting: structured-output requests whose
        # grammar was not yet ready get placed here by the base scheduler.
        # We must route them through holdback to enforce the
        # one-prefill-at-a-time constraint.
        while self.skipped_waiting:
            holdback_queue.append(self.skipped_waiting.pop_request())

        # req_id -> cached_blocks, new_blocks
        required_blocks = dict[str, tuple[int, int]]()

        # Check if new requests can be scheduled for prefill
        available_blocks = self._get_free_blocks() - self.total_reserved_blocks
        while holdback_queue:
            new_request = holdback_queue[0]
            cached, blocks = self._get_required_blocks(new_request, True)
            if blocks > available_blocks:
                break

            if self.can_schedule_prefill(new_request):
                holdback_queue.popleft()
                required_blocks[new_request.request_id] = (cached, blocks)
                available_blocks -= blocks

                logger.debug(
                    "Scheduling a new request (%d prompt tokens), holding back %d requests",
                    new_request.num_prompt_tokens,
                    len(holdback_queue),
                )

                # Add request to the waiting queue
                self.waiting.append(new_request)
            else:
                # Otherwise, we simply stop here so that the scheduler
                # can work with the batch we have
                break

        assert len(self.ongoing_prefills) <= 1, (
            "Only one request can be prefilled at a time, but got %d" % len(self.ongoing_prefills)
        )
        assert len(self.waiting) == 0 or len(self.ongoing_prefills) == 0, (
            "Cannot schedule new requests while another request prefill is ongoing."
        )
        assert all(r in self.running for r in self.ongoing_prefills), (
            "Ongoing prefill requests must be in the running queue."
        )

        new_prefill_candidates: list[Request] = []

        # Check ongoing prefills
        if self.ongoing_prefills:
            # Some running requests are currently being prefilled. We need to
            # separate them from currently decoding requests, and schedule
            # them separately. Either we schedule a chunked prefill step, or a
            # decoding step

            assert len(self.ongoing_prefills) == 1

            schedule_prefill = self.can_schedule_prefill(self.ongoing_prefills[0])

            if schedule_prefill:
                running_holdback = [r for r in self.running if r not in self.ongoing_prefills]
                self.running = self.ongoing_prefills
                self.previous_step_was_prefill = True
            else:
                self.running = [r for r in self.running if r not in self.ongoing_prefills]
                running_holdback = self.ongoing_prefills
                self.previous_step_was_prefill = False

        # Check new requests to prefill
        elif len(self.waiting) > 0:
            # Try to promote grammar-waiting requests whose FSM is now
            # ready, so we correctly classify ready vs not-ready requests.
            for r in list(self.waiting):
                if r.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
                    so_req = r.structured_output_request
                    if so_req and so_req.grammar:
                        r.status = RequestStatus.WAITING

            ready_to_prefill = [
                r
                for r in self.waiting
                if r.status != RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
            ]
            if ready_to_prefill:
                new_prefill_candidates = list(self.waiting)
                # Hide current decodes from the scheduler
                running_holdback = self.running
                self.running = []
                self.previous_step_was_prefill = True
            else:
                # Grammar not yet initialized for any waiting request.
                # Return them to holdback so the base scheduler doesn't
                # try to promote and schedule them alongside decodes.
                while self.waiting:
                    holdback_queue.appendleft(self.waiting.pop())
                running_holdback = []
                self.previous_step_was_prefill = False
        else:
            self.previous_step_was_prefill = False
            running_holdback = []

        # Cap chunk-0 token count to chunk_size - left_padding so the upstream KV
        # cache manager doesn't allocate a real blocks for the left-padding region.
        # Only matters at chunk 0; later chunks land on natural chunk boundaries.
        # Mutating scheduler_config is safe: the SpyreScheduler is the only
        # scheduler in this engine and at most one prefill is in flight per step.
        self.scheduler_config.long_prefill_token_threshold = self._current_chunk_token_threshold(
            new_prefill_candidates
        )

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # Track as ongoing prefills only the requests that were actually
        # scheduled (i.e., moved from waiting to running by the base
        # scheduler).
        if new_prefill_candidates:
            self.ongoing_prefills.extend(r for r in new_prefill_candidates if r in self.running)

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        # Log the scheduled tokens not at every step, but when doing chunked
        # prefill. These include decode steps during interleaving
        if self.ongoing_prefills or any(
            r.num_computed_tokens <= r.num_prompt_tokens + 1 for r in self.running
        ):
            logger.debug("Scheduled tokens in this step: %s", outputs.num_scheduled_tokens)

        # Collect grammar bitmask synchronously for structured outputs.
        # NOTE: This is done here because vllm-spyre currently combines token sampling
        # in model_executor.execute_model() rather than implementing sample_tokens()
        # in the model runner. This means we cannot collect the grammar bitmask
        # asynchronously while the model is running (as done in vLLM core).
        # TODO: Implement sample_tokens() in SpyreModelRunner to enable async grammar
        # collection for better performance.
        outputs._spyre_grammar_output = self.get_grammar_bitmask(outputs)  # type: ignore[attr-defined]

        # As blocks are allocated, we discount them from the reserved blocks.
        # For prefill blocks we must first subtract the cached blocks.
        free_blocks = self._get_free_blocks()
        for new_request in outputs.scheduled_new_reqs:
            cached, reserved = required_blocks[new_request.req_id]
            scheduled_blocks = len(new_request.block_ids[0])
            new_blocks = scheduled_blocks - cached
            # The first chunk of a prefill that is scheduled
            # always has at least one new block
            assert new_blocks >= 1
            actual_reserved = reserved - new_blocks
            assert actual_reserved >= 0
            self.total_reserved_blocks += actual_reserved
            self.reserved_blocks[new_request.req_id] = actual_reserved

        for req_id, req_new_blocks in zip(
            outputs.scheduled_cached_reqs.req_ids,
            outputs.scheduled_cached_reqs.new_block_ids,
        ):
            new_blocks = 0 if req_new_blocks is None else len(req_new_blocks[0])
            self.total_reserved_blocks -= new_blocks
            self.reserved_blocks[req_id] -= new_blocks
            assert self.reserved_blocks[req_id] >= 0

        assert 0 <= self.total_reserved_blocks <= free_blocks

        # Inject step-start timestamps for timing measurement
        if self._bench is not None:
            now = time.time()
            if self.previous_step_was_prefill:
                self._bench.prefill_step_start = now
                self._bench.decode_step_start = None
            else:
                self._bench.decode_step_start = now
                self._bench.prefill_step_start = None

        return outputs

    def can_schedule_prefill(self, request: Request) -> bool:
        # running and waiting queues are both empty, we can start a new batch
        # which can always be scheduled
        if len(self.running) + len(self.waiting) == 0:
            return True

        if not self._has_scheduling_priority(request):
            return False

        return self._satisfies_constraints(request)

    def _satisfies_constraints(self, request: Request) -> bool:
        # Use a local variable to check the prefix cache hit length ahead of time without mutating
        # request.num_computed_tokens
        num_computed_tokens = request.num_computed_tokens
        if num_computed_tokens == 0:
            # NB: self.kv_cache_manager comes from the parent class, and we are being super nosy.
            # This update ensures that we know when we're scheduling the last prefix chunk, in the
            # case where most of the prompt hits prefix cache and we only run a single chunk.
            _, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)

        is_first_chunk = request.num_computed_tokens == 0
        is_last_chunk = (request.num_prompt_tokens - num_computed_tokens) <= self.chunk_size

        if not self.do_interleaving:
            # All the prefills are consecutive, so the first chunk has to
            # satisfy all the constraints, and we don't need to check them again
            # for subsequent chunks.
            if not is_first_chunk:
                return True

            return self._satisfies_first_chunk_constraints(
                request
            ) and self._satisfies_last_chunk_constraints(request)

        can_schedule = True
        if is_first_chunk:
            can_schedule = self._satisfies_first_chunk_constraints(request)

        if is_last_chunk:
            can_schedule = can_schedule and self._satisfies_last_chunk_constraints(request)

        return can_schedule

    def _satisfies_first_chunk_constraints(self, request: Request) -> bool:
        """First chunked prefill can be scheduled only if there is space in the
        input batch (cond1) and in the prefill batch (cond2)."""

        # TODO theoretically we could already do a chunked prefill even
        # if the decode batch is full, but the current implementation of input
        # batch doesn't allow to do so.
        num_running = len(self.running)
        cond1 = num_running + len(self.waiting) < self.max_num_running_reqs

        # check that there is space in the prefill batch
        max_prefill_batch_size = 1
        cond2 = len(self.waiting) < max_prefill_batch_size

        return cond1 and cond2

    def _satisfies_last_chunk_constraints(self, request: Request) -> bool:
        """Last chunked prefill can be scheduled only if there is enough space
        in the decode batch, and if all the other spyre-related conditions
        are satisfied."""
        decoding_requests = [r for r in self.running if r not in self.ongoing_prefills]

        # check that there is space in the current decode batch
        num_running = len(decoding_requests)
        cond1 = num_running + len(self.waiting) < self.max_num_running_reqs

        # calculate new max tkv of the batch given the new sequence joins
        # considers all possible cases:
        # - prompt_len > self.tkv and fall into different blocks
        # - prompt_len and self.tkv fall within the same block
        # - prompt_len < self.tkv and fall into different blocks
        prompt_len = request.num_prompt_tokens
        n_blocks = math.floor(max(self.tkv, prompt_len) / self.block_size)
        new_req_tkv = n_blocks * self.block_size + prompt_len % self.block_size

        # check that batch size x tkv is smaller than the max supported number
        # Note: using max_tkv is a conservative upper bound here. For the
        # optimal check we need model runner to return per sequence tkvs
        cond2 = lambda: self.check_batch_tkv_limit_cp(
            request=request,
            new_req_tkv=new_req_tkv,
            running=decoding_requests,
        )

        return cond1 and cond2()

    def _has_scheduling_priority(self, request):
        decoding_requests = [r for r in self.running if r not in self.ongoing_prefills]

        # If we do interleaving, then two consecutive prefill steps are
        # forbidden when there are decoding requests
        if self.do_interleaving and self.previous_step_was_prefill and len(decoding_requests) > 0:
            return False

        # Requests that are already prefilling are prioritized over new requests
        if request in self.ongoing_prefills:
            return True

        # We can start prefilling a new requests if we satisfy the maximum
        # number of concurrent prefills
        max_concurrent_prefills = 1
        num_prefills = len(self.waiting) + len(self.ongoing_prefills)
        return num_prefills < max_concurrent_prefills

    def check_batch_tkv_limit_cp(self, request: Request, new_req_tkv: int, running) -> bool:
        """
        Check whether adding a new sequence to the decode batch would violate
        Spyre's maximum batch volume constraint for chunked prefill.

        In Spyre, the product of `batch_size` and the current `tkv`
        (tokens-per-sequence) must not exceed the limit defined by
        `VLLM_DT_MAX_BATCH_TKV_LIMIT`. Before scheduling a new sequence,
        we must ensure that this constraint will hold for all decoding
        steps that result from combining the new sequence with the currently
        running decode batch.

        This implementation:
        1. Computes the maximum possible `tkv` for each sequence in the
        decode batch.
        2. Sorts these values in ascending order.
        3. Iterates through them, stopping once the `tkv` of the new sequence.
        is reached. Remaining sequences do not need to be checked explicitly,
        since they were validated when they were added (by inductive reasoning).

        Note: drawing explaining the algorithm in more detail uploaded here:
        https://github.com/torch-spyre/sendnn-inference/pull/363#issuecomment-3173605517
        """

        # Compute the effective token length of the new request
        # Rounded up to the nearest block size to account for potential padding
        new_req_max_tkv = round_up_to_block_size(new_req_tkv + request.max_tokens - 1)
        # Extra block of slack: left-padding can push a sequence's runtime tkv up to
        # one block past the scheduler's estimate when the batch re-aligns on admission.
        new_req_max_tkv += self.block_size

        # Compute token lengths for all running requests (decode batch)
        decode_req_max_tkvs = []
        # Decide new tkv based on max of current tkv or new request prompt tokens
        dec_req_tkv = max(self.tkv, request.num_prompt_tokens)
        for req in running:
            n_generated_output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            # Rounded up to the nearest block size to account for potential padding
            dec_req_max_tkv = round_up_to_block_size(
                dec_req_tkv + (req.max_tokens - n_generated_output_tokens) - 1
            )
            # Extra block of slack: left-padding can push a sequence's runtime tkv up to
            # one block past the scheduler's estimate when the batch re-aligns on admission.
            dec_req_max_tkv += self.block_size

            decode_req_max_tkvs.append(dec_req_max_tkv)

        # Sort decode requests token lengths in ascending order
        decode_req_max_tkvs.sort()

        # Initialize values
        # The request is already in the running queue if it has done a first
        # chunked prefill
        batch_size = len(running)
        if request not in running:
            batch_size += 1
        max_batch_tkv = 0

        # Try adding the new request to the batch and check the max volume
        for decode_req_max_tkv in decode_req_max_tkvs:
            if new_req_max_tkv <= decode_req_max_tkv:
                # If the new request is shorter, it limits the batch volume
                max_batch_tkv = max(max_batch_tkv, batch_size * new_req_max_tkv)
                break
            else:
                # Otherwise, use the current (longer) request's volume
                max_batch_tkv = max(max_batch_tkv, batch_size * decode_req_max_tkv)
                # decrease batch_size by 1 as the current request finished
                batch_size -= 1

        return max_batch_tkv <= self.max_batch_tkv_limit

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str], None],
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """Handles removing finished requests from ongoing_prefills"""
        if isinstance(request_ids, str):
            request_ids = (request_ids,)

        # first defer to vLLM scheduler
        # validates the input requests and generates the output
        aborted_requests = super(SpyreScheduler, self).finish_requests(
            request_ids=request_ids, finished_status=finished_status
        )

        # request_ids None means all requests are finished
        self.ongoing_prefills = (
            []
            if request_ids is None
            else [r for r in self.ongoing_prefills if r.request_id not in request_ids]
        )

        return aborted_requests

    def calc_cached_tokens(self, prompt_len: int) -> tuple[int, int]:
        blocks_per_chunk = self.chunk_size // self.block_size
        n_chunks = math.ceil(prompt_len / self.chunk_size)
        n_blocks = math.ceil(prompt_len / self.block_size)

        total_blocks = n_chunks * blocks_per_chunk
        n_padding_tokens = (total_blocks - n_blocks) * self.block_size
        total_cached_toks = (prompt_len // self.chunk_size) * self.chunk_size
        return max(0, total_cached_toks - n_padding_tokens), n_padding_tokens

    def adjust_hit(self, prompt_len: int, hit: int):
        assert hit % self.block_size == 0

        max_possible, padding = self.calc_cached_tokens(prompt_len)

        if hit >= max_possible:
            return max_possible

        # if the hit is in the middle of a chunk, we also need to discard that chunk
        actual_hit = max(0, (((padding + hit) // self.chunk_size) * self.chunk_size) - padding)
        return actual_hit

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        """Update the scheduler stats from the base scheduler.
        In sendnn-inference the last chunk is always recomputed, even though
        the space is not duplicated.
        Spyre does not support cross-request MM cache reuse today, so MM cache
        hit reporting is forced to 0.0%.
        """
        base_stats = super().make_stats(*args, **kwargs)

        if base_stats is not None and base_stats.prefix_cache_stats is not None:
            base_stats.prefix_cache_stats.hits = self.adjust_hit(
                base_stats.prefix_cache_stats.queries, base_stats.prefix_cache_stats.hits
            )

        if base_stats is not None:
            mm_cache_stats = getattr(base_stats, "mm_cache_stats", None)
            if mm_cache_stats is not None:
                mm_cache_stats.hits = 0

        return base_stats
