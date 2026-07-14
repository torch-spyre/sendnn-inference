# SPDX-License-Identifier: Apache-2.0

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Union
from collections import defaultdict


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
class MMEncodeRequest:
    """Lightweight descriptor for a waiting MM request that should be
    pre-encoded before its Spyre prefill step begins."""

    request_id: str
    prompt_token_ids: list[int]
    mm_features: list = field(default_factory=list)


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

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
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
        outputs = super(SpyreScheduler, self).schedule(throttle_prefills=throttle_prefills)

        # first move skipped and then unscheduled requests back
        # to the waiting queue, preserving priority
        while skip_queue:
            self.waiting.append(skip_queue.popleft())

        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

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


@dataclass
class ChunkedPrefillSpyreSchedulerStats:
    decode_batch_size: int = 0
    num_paused_reqs: int = 0
    pause_events: int = 0
    resume_events: int = 0


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

        - Volumetric constraint: the product of batch_size and current TKV
            must not exceed `VLLM_DT_MAX_BATCH_TKV_LIMIT` when adding a new
            request. See `_can_decode_all_requests()` method for details.

        - Decode pausing: requests may be temporarily paused from decoding
            when the batch TKV limit would be exceeded in the next decode step.
            Paused requests are resumed when capacity becomes available.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.chunk_size = self.scheduler_config.max_num_batched_tokens

        # We want to keep track of requests for which the prefill is ongoing.
        # Theoretically, only one request can be prefilled at a time, but we
        # keep a list to be able to batch prefills in the future.
        self.ongoing_prefills: list[Request] = []

        # Track requests that were temporarily paused from decoding due to
        # batch TKV constraint and moved back to waiting queue
        self.paused_decoding_requests: list[Request] = []

        # Prefills interleaving: if the feature flag is set, prefill operations
        # are interleaved with a decode step. This allows to minimize currently
        # decoding requests
        self.do_interleaving: bool = envs_spyre.SENDNN_INFERENCE_CP_INTERLEAVE_STEPS
        self.step_is_prefill: bool = False

        self.tkv = 0
        self.block_size = SpyrePlatform.get_block_size()

        # Async MM encoding state.
        # _mm_encoding_submitted: requests whose encode job has been dispatched to
        #   the encoder subprocess but whose result has not yet been received.
        # _mm_encoding_ready: requests whose embeddings are ready in
        #   pending_mm_embeddings (confirmed via _spyre_newly_encoded_req_ids in
        #   the model runner output).  Only MM requests in this set are eligible
        #   for prefill scheduling.
        self._mm_encoding_submitted: set[str] = set()
        self._mm_encoding_ready: set[str] = set()
        self.max_batch_tkv_limit = SpyrePlatform.get_max_batch_tkv_limit()

        assert self.max_batch_tkv_limit != -1, (
            "Expecting the env var VLLM_DT_MAX_BATCH_TKV_LIMIT to be set in platform.py"
        )

        self.total_reserved_blocks = 0
        self.reserved_blocks = dict[str, int]()
        self.pause_events = 0
        self.resume_events = 0

        self.request_last_decode_step = defaultdict(int)
        self.long_output_prio = envs_spyre.SENDNN_INFERENCE_LONG_OUT_PRIO

    def update_from_output(self, scheduler_output, model_runner_output):
        assert isinstance(model_runner_output, SpyreModelRunnerOutput), (
            "Expecting an instance of CPSpyreModelRunnerOutput when doing chunked prefill."
        )

        # Update async MM encoding state: move newly encoded requests from
        # "submitted" to "ready" so they become eligible for prefill.
        # Read from scheduler_output (set by SpyreMultiprocExecutor.execute_model)
        # rather than model_runner_output — the executor uses non_block=True which
        # returns a Future, so attributes set on the Future never reach the resolved
        # ModelRunnerOutput.  scheduler_output is the same object in both places.
        for req_id in getattr(scheduler_output, "_spyre_newly_encoded_req_ids", []):
            self._mm_encoding_submitted.discard(req_id)
            # Only promote to ready if the request is still known to the scheduler.
            # If it was aborted while encoding was in-flight, finish_requests already
            # removed it — skip to avoid a stale _mm_encoding_ready entry.
            if req_id in self.requests:
                self._mm_encoding_ready.add(req_id)
        # Abort any request whose encode job failed — no retries.
        for req_id in getattr(scheduler_output, "_spyre_failed_encode_req_ids", []):
            logger.error("MM encode failed for req '%s' — aborting request", req_id)
            self.finish_requests([req_id], RequestStatus.FINISHED_ABORTED)

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
            self.request_last_decode_step.pop(finished_request, None)

        return result

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

    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
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

        # Check if new requests can be scheduled for prefill.
        # Per-request ineligibility (MM encoding not ready, shape mismatch, …)
        # is collected in skipped_requests and restored to holdback_queue after
        # the loop so they remain available for future schedule() calls.
        # Block-count capacity is the only FIFO-correct reason to stop early:
        # if the front request exceeds available blocks, later requests are
        # unlikely to fit either.
        available_blocks = self._get_free_blocks() - self.total_reserved_blocks
        skipped_requests: list[Request] = []
        while holdback_queue:
            new_request = holdback_queue.popleft()
            cached, blocks = self._get_required_blocks(new_request, True)
            if blocks > available_blocks:
                holdback_queue.appendleft(new_request)
                break

            if self.can_schedule_prefill(new_request):
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
                # Per-request reason (e.g. MM encoding still in-flight): skip
                # this request and keep checking later ones.
                skipped_requests.append(new_request)

        # Restore skipped requests at the front of holdback_queue so they
        # are returned to self.waiting (line below) in their original order.
        for req in reversed(skipped_requests):
            holdback_queue.appendleft(req)

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
                self.step_is_prefill = True
            else:
                self.running = [r for r in self.running if r not in self.ongoing_prefills]
                running_holdback = self.ongoing_prefills
                self.step_is_prefill = False

        # Check new requests to prefill
        elif len(self.waiting) > 0:
            # Promote any requests whose structured-output grammar has become
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
                new_prefill_candidates = ready_to_prefill
                # Hide current decodes from the scheduler
                running_holdback = self.running
                self.running = []
                self.step_is_prefill = True
            else:
                # All waiting requests are still blocked on grammar; put them
                # back in holdback so the base scheduler doesn't see them
                # alongside decodes.
                while self.waiting:
                    holdback_queue.appendleft(self.waiting.pop())
                running_holdback = []
                self.step_is_prefill = False
        else:
            self.step_is_prefill = False
            running_holdback = []

        if not self.step_is_prefill:
            self._handle_decode_requests_pausing()

        # Collect MM encode requests for ALL waiting multimodal requests that
        # have not yet been submitted to the encoder subprocess.  Emitted on
        # every schedule() call (prefill AND decode steps) so the encoder can
        # stay ahead of the prefill queue.  The executor submits each request
        # exactly once (tracked here via _mm_encoding_submitted).
        mm_encode_requests: list[MMEncodeRequest] = []
        for req in holdback_queue:
            if not getattr(req, "mm_features", None):
                continue
            if req.request_id in self._mm_encoding_submitted:
                continue
            if req.request_id in self._mm_encoding_ready:
                continue
            mm_encode_requests.append(
                MMEncodeRequest(
                    request_id=req.request_id,
                    prompt_token_ids=list(req.prompt_token_ids or []),
                    mm_features=req.mm_features,
                )
            )
            self._mm_encoding_submitted.add(req.request_id)
            if len(mm_encode_requests) >= self.max_num_running_reqs:
                break

        # Cap chunk-0 token count to chunk_size - left_padding so the upstream KV
        # cache manager doesn't allocate a real blocks for the left-padding region.
        # Only matters at chunk 0; later chunks land on natural chunk boundaries.
        # Mutating scheduler_config is safe: the SpyreScheduler is the only
        # scheduler in this engine and at most one prefill is in flight per step.
        self.scheduler_config.long_prefill_token_threshold = self._current_chunk_token_threshold(
            new_prefill_candidates
        )

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule(throttle_prefills=throttle_prefills)

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

        outputs._spyre_mm_encode_requests = mm_encode_requests  # type: ignore[attr-defined]

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
        return outputs

    def can_schedule_prefill(self, request: Request) -> bool:
        # MM requests must wait until their vision embedding is ready.
        # Only applies in async encoder mode; in non-async mode nothing ever
        # populates _mm_encoding_ready so the gate would block all MM requests.
        # Text-only requests are completely unaffected by this check.
        if getattr(request, "mm_features", None) and (
            envs_spyre.SENDNN_INFERENCE_ASYNC_MM_ENCODER
            and request.request_id not in self._mm_encoding_ready
        ):
            return False

        # running and waiting queues are both empty, we can start a new batch
        # which can always be scheduled
        if len(self.running) + len(self.waiting) == 0:
            return True

        # Paused request have the priority and will be resumed if the tkv_batch limit allows it
        if self.paused_decoding_requests:
            return False

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

        # NOTE: We could already do a chunked prefill even if the decode batch
        # is full, this could potentially increase the ITL of the request
        # if it then request doesn't satisfy the volumetric constraint
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

        if cond1 and envs_spyre.SENDNN_INFERENCE_PAUSING_ENABLED:
            return True

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
        if self.do_interleaving and self.step_is_prefill and len(decoding_requests) > 0:
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

    def _can_decode_all_requests(self, decoding_requests: list[Request]) -> bool:
        """
        Check if all decoding requests can be decoded in the next step without
        violating the max batch TKV limit.
        """
        if not decoding_requests:
            return True

        next_predicted_tkv = self.predict_next_decode_tkv(decoding_requests)

        # the tkv should never get beyond max_model_len
        assert next_predicted_tkv <= self.max_model_len

        # check batch tkv limit: batch_size * predicted_tkv must not exceed limit
        batch_size = len(decoding_requests)
        predicted_batch_tkv = batch_size * next_predicted_tkv

        return predicted_batch_tkv <= self.max_batch_tkv_limit

    def _handle_decode_requests_pausing(self) -> None:
        decoding_requests = [r for r in self.running if r not in self.ongoing_prefills]
        requests_by_step = list[tuple[Request, int]]()

        was_paused = set[str]()
        was_running = set[str]()

        for req in self.paused_decoding_requests:
            requests_by_step.append((req, self.request_last_decode_step[req.request_id]))
            was_paused.add(req.request_id)

        for req in decoding_requests:
            requests_by_step.append((req, self.request_last_decode_step[req.request_id]))
            was_running.add(req.request_id)

        # Sort is stable, so requests with the same last
        # step will be sorted by how many tokens they have already generated
        request_order = sorted(
            requests_by_step,
            key=lambda x: x[0].num_computed_tokens - x[0].num_prompt_tokens,
            reverse=self.long_output_prio,
        )
        request_order.sort(key=lambda x: x[1], reverse=True)

        self.paused_decoding_requests.clear()
        decoding_requests.clear()

        for req, _ in request_order:
            if self._can_decode_all_requests(decoding_requests + [req]):
                decoding_requests.append(req)
                self.request_last_decode_step[req.request_id] = 0
                if req.request_id in was_paused:
                    self.running.append(req)
            else:
                self.paused_decoding_requests.append(req)
                self.request_last_decode_step[req.request_id] += 1
                if req.request_id in was_running:
                    self.running.remove(req)

        pause_inc = len(self.paused_decoding_requests) - len(was_paused)
        if pause_inc >= 0:
            self.pause_events += pause_inc
        else:
            self.resume_events -= pause_inc

    def predict_next_decode_tkv(self, running_requests: list[Request]) -> int:
        """
        Predicts the TKV after the next decode step for a given batch of running
        requests.

        This method replicates the TKV calculation logic from the model runner's
        _prepare_decode method, accounting for:
        - Block alignment (left-padding to make batch rectangular)
        - The next token that will be generated (+1)
        - Maximum TKV across all requests in the batch

        Args:
            running_requests: List of Request objects currently in the decode batch

        Returns:
            The predicted TKV value after the next decode step
        """
        if not running_requests:
            return 0

        # Step 1: Find the maximum number of blocks across all requests
        # Account for requests that will need a new block after the next token
        max_n_blocks = 0
        num_blocks_per_req: list[int] = []
        for request in running_requests:
            num_blocks = math.ceil((request.num_computed_tokens + 1) / self.block_size)
            num_blocks_per_req.append(num_blocks)
            max_n_blocks = max(max_n_blocks, num_blocks)

        # Step 2: Calculate TKV for each request and find the maximum
        max_tkv = 0
        for request, num_blocks in zip(running_requests, num_blocks_per_req):
            # Calculate left padding blocks needed for alignment
            left_pad_blocks_count = max_n_blocks - num_blocks
            left_padding = left_pad_blocks_count * self.block_size

            # Calculate TKV for this request (including the next token)
            req_tkv = left_padding + request.num_computed_tokens + 1

            # Track the maximum TKV
            max_tkv = max(max_tkv, req_tkv)

        return max_tkv

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str], None],
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """
        Handles removing finished requests from ongoing_prefills and
        paused_decoding_requests
        """
        if isinstance(request_ids, str):
            request_ids = (request_ids,)

        # first defer to vLLM scheduler
        # validates the input requests and generates the output
        aborted_requests = super(SpyreScheduler, self).finish_requests(
            request_ids=request_ids, finished_status=finished_status
        )

        # request_ids None means all requests are finished
        if request_ids is None:
            self.ongoing_prefills = []
            self._mm_encoding_submitted.clear()
            self._mm_encoding_ready.clear()
        else:
            self.ongoing_prefills = [
                r for r in self.ongoing_prefills if r.request_id not in request_ids
            ]
            for rid in request_ids:
                # If the encode job is queued but not yet started, send a cancel
                # token so the encoder subprocess skips it rather than running a
                # full (expensive) vision-tower forward for a dead request.
                if rid in self._mm_encoding_submitted:
                    from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

                    cq = SpyreMultiprocExecutor.get_mm_cancel_queue()
                    if cq is not None:
                        try:
                            cq.put_nowait(rid)
                        except Exception as exc:
                            logger.debug(
                                "scheduler: failed to send cancel for req '%s': %s",
                                rid,
                                exc,
                            )
                self._mm_encoding_submitted.discard(rid)
                self._mm_encoding_ready.discard(rid)

        # Also remove from paused_decoding_requests
        self.paused_decoding_requests = (
            []
            if request_ids is None
            else [r for r in self.paused_decoding_requests if r.request_id not in request_ids]
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

        if base_stats is not None:
            if base_stats.prefix_cache_stats is not None:
                base_stats.prefix_cache_stats.hits = self.adjust_hit(
                    base_stats.prefix_cache_stats.queries, base_stats.prefix_cache_stats.hits
                )

                mm_cache_stats = getattr(base_stats, "mm_cache_stats", None)
                if mm_cache_stats is not None:
                    mm_cache_stats.hits = 0

            decode_batch_size = sum(1 for r in self.running if r not in self.ongoing_prefills)
            num_paused_reqs = len(self.paused_decoding_requests)

            if base_stats.kv_connector_stats is None:
                base_stats.kv_connector_stats = {}

            # Abuse the kv_connector_stats field to store the spyre stats.
            # We can open an upstream PR to add another extensible field.
            base_stats.kv_connector_stats["sendnn-stats"] = ChunkedPrefillSpyreSchedulerStats(
                decode_batch_size=decode_batch_size,
                num_paused_reqs=num_paused_reqs,
                pause_events=self.pause_events,
                resume_events=self.resume_events,
            )
            self.pause_events = 0
            self.resume_events = 0

        return base_stats
