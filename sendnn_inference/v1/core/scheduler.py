# SPDX-License-Identifier: Apache-2.0

import math
from collections import deque
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
        self.previous_step_was_prefill: bool = False

        self.tkv = 0
        self.block_size = SpyrePlatform.get_block_size()
        self.max_batch_tkv_limit = SpyrePlatform.get_max_batch_tkv_limit()

        assert self.max_batch_tkv_limit != -1, (
            "Expecting the env var VLLM_DT_MAX_BATCH_TKV_LIMIT to be set in platform.py"
        )

        self.total_reserved_blocks = 0
        self.reserved_blocks = dict[str, int]()

    def update_from_output(self, scheduler_output, model_runner_output):
        assert isinstance(model_runner_output, SpyreModelRunnerOutput), (
            "Expecting an instance of CPSpyreModelRunnerOutput when doing chunked prefill."
        )

        # Update the correct num_computed_tokens value given left-padding and
        # prefix cache hit info
        for req in self.ongoing_prefills:
            # The number of computed tokens only need to be adapted when it is
            # the first chunk of a multi-chunk prefill
            is_first_chunk = req.num_computed_tokens <= self.chunk_size
            is_last_chunk = req.num_computed_tokens == req.num_prompt_tokens
            if is_first_chunk and not is_last_chunk:
                left_padding = model_runner_output.left_padding.get(req.request_id, 0)
                prefix_cache_len = model_runner_output.prefix_cache_hit_len.get(req.request_id, 0)

                req.num_computed_tokens = self.adjust_computed_tokens(
                    computed_tokens=req.num_computed_tokens,
                    left_padding=left_padding,
                    prefix_cache_len=prefix_cache_len,
                )

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

    def adjust_computed_tokens(
        self, computed_tokens: int, left_padding: int, prefix_cache_len: int
    ) -> int:
        """
        Returns an adjusted `num_computed_tokens` given left padding and prefix
        cache hit info.
        """
        # The prefix cache length is already adjusted for left padding.
        # If it's bigger than the number of computed tokens, then we hit more
        # prefix cache than we scheduled.
        if prefix_cache_len > computed_tokens:
            assert (prefix_cache_len + left_padding) % self.chunk_size == 0
            return prefix_cache_len
        # Otherwise just account for the left padding
        return computed_tokens - left_padding

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
                if r.status != RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR  # type: ignore[attr-defined]
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

        if not self.previous_step_was_prefill:
            self._handle_decode_requests_pausing()

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

        return outputs

    def can_schedule_prefill(self, request: Request) -> bool:
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

        # Check that the current decode batch is not about to have requests paused.
        # This avoids adding more request to be paused and seems to slightly improve
        # metrics.
        cond2 = lambda: self._can_decode_all_requests(self.running)

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
        """
        Manage pausing and resuming of decode requests based on batch TKV constraints.

        This method:
        1. Pauses requests with the fewest decoded tokens when batch TKV limit is exceeded
        2. Resumes previously paused requests (oldest first) when capacity is available
        """
        decoding_requests = [r for r in self.running if r not in self.ongoing_prefills]

        had_to_remove = False
        initial_had_requests = len(decoding_requests) > 0

        # If we can't decode all requests due to batch TKV limits, iteratively
        # remove requests with the fewest decoded tokens and pause them until
        # the remaining batch fits within constraints
        while not self._can_decode_all_requests(decoding_requests):
            had_to_remove = True

            # TODO we should test different removal logics: longest request, optimize padding
            # Remove the request with the fewest decoded tokens
            # Decoded tokens = num_computed_tokens - num_prompt_tokens
            request_to_remove = min(
                decoding_requests, key=lambda r: r.num_computed_tokens - r.num_prompt_tokens
            )
            decoding_requests.remove(request_to_remove)
            self.running.remove(request_to_remove)
            self.paused_decoding_requests.append(request_to_remove)
            logger.info("Request %s paused due to batch TKV limit ", request_to_remove.request_id)

        # It shouldn't be possible to remove all requests if we started with some
        assert not initial_had_requests or len(decoding_requests) > 0

        # If we didn't have to remove any requests, try to add back previously
        # paused requests (oldest first) as long as they fit within constraints
        if not had_to_remove:
            while self.paused_decoding_requests:
                # Try adding the oldest paused request (first in list)
                request_to_add = self.paused_decoding_requests[0]
                test_requests = decoding_requests + [request_to_add]

                if self._can_decode_all_requests(test_requests):
                    # Can add this request back
                    self.paused_decoding_requests.pop(0)
                    self.running.append(request_to_add)
                    decoding_requests.append(request_to_add)
                    logger.info(
                        "Request %s resumed (batch TKV capacity available).",
                        request_to_add.request_id,
                    )
                else:
                    # Can't add any more requests
                    break

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
        self.ongoing_prefills = (
            []
            if request_ids is None
            else [r for r in self.ongoing_prefills if r.request_id not in request_ids]
        )

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

        if base_stats is not None and base_stats.prefix_cache_stats is not None:
            base_stats.prefix_cache_stats.hits = self.adjust_hit(
                base_stats.prefix_cache_stats.queries, base_stats.prefix_cache_stats.hits
            )

        if base_stats is not None:
            mm_cache_stats = getattr(base_stats, "mm_cache_stats", None)
            if mm_cache_stats is not None:
                mm_cache_stats.hits = 0

        return base_stats
