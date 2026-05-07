# SPDX-License-Identifier: Apache-2.0
"""
Spyre scheduler classes.

This module provides Spyre-specific scheduler subclasses of vLLM's
``Scheduler``.  An async-scheduling variant for chunked prefill (subclassing
``AsyncScheduler``) lives in ``async_scheduler.py``.
"""

import math
from collections import deque
from typing import Iterable, Union

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request, RequestStatus

import sendnn_inference.envs as envs_spyre
from sendnn_inference.platform import SpyrePlatform
from sendnn_inference.v1.worker.spyre_model_runner import SpyreModelRunnerOutput

logger = init_logger(__name__)

# Ensure that block_size is 64
# This ensures the rounding function is correct
assert SpyrePlatform.get_block_size() == 64


def round_up_to_block_size(n: int) -> int:
    # Helper function to round up to the nearest block size
    # Uses bitwise alignment for better performance
    return (n + 63) & ~63


class PoolingSpyreScheduler(Scheduler):
    """Scheduler that adds Spyre warmup-shape constraints for pooling models."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config
        self.spyre_warmup_shapes: tuple[dict[str, int], ...] = SpyrePlatform.get_warmup_shapes(
            self.scheduler_config
        )

    def schedule(self) -> SchedulerOutput:
        """Add Spyre warmup-shape constraints then delegate to the base scheduler."""
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

        outputs = super().schedule()

        # first move skipped and then unscheduled requests back
        # to the waiting queue, preserving priority
        while skip_queue:
            self.waiting.append(skip_queue.popleft())

        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        outputs._spyre_grammar_output = self.get_grammar_bitmask(outputs)
        return outputs

    def _get_matching_warmup_shapes(
        self, request: Request, warmup_shapes: list[dict[str, int]], current_batch_size: int
    ) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request."""
        return [
            shape
            for shape in warmup_shapes
            if request.num_prompt_tokens <= shape["prompt_length"]
            and current_batch_size < shape["batch_size"]
        ]


class ChunkedPrefillSpyreScheduler(Scheduler):
    """Scheduler that adds Spyre chunked-prefill constraints.

    The async-scheduling variant ``AsyncChunkedPrefillSpyreScheduler`` (in
    ``async_scheduler.py``) subclasses this class and ``AsyncScheduler`` so
    that ``super().schedule()`` resolves to ``AsyncScheduler.schedule()`` via
    the MRO.

    Spyre scheduling policy
    -----------------------
    Prefill vs decode priority:
        - A new request cannot start prefill while another request's prefill
          is on-going.
        - Prefill steps are optionally interleaved with one decode step.
        - Prefill has priority over decode (subject to the interleaving
          constraint).
        - If a prefill step is prevented by constraints a decode step is
          scheduled instead.

    Spyre constraints (applied at the last chunk of a chunked prefill):
        - Prefill batch size: only one request's chunked prefill at a time.
        - Decode batch size: at most max_num_seqs running requests.
        - Max model length: requested tokens must fit the context window.
        - Volumetric: batch_size × tkv ≤ VLLM_DT_MAX_BATCH_TKV_LIMIT.

    Under async scheduling, the optimistic ``num_computed_tokens`` advance
    performed by the base scheduler is recorded per-request in
    ``_inflight_prefill_tokens`` on each prefill chunk;
    ``update_from_output()`` reconciles the advance against the runner's
    actual left-padding / prefix-cache report and decrements the counter.
    The same code path works for sync mode where the counter is added and
    cleared within a single step (a no-op). ``update_from_output()`` also
    filters scheduler outputs to only include requests that were actually
    executed (relevant under run-ahead).

    Note: we deliberately do NOT mirror the prefill advance into upstream's
    ``Request.num_output_placeholders``; that field feeds upstream's
    ``num_new_tokens = num_tokens_with_spec + num_output_placeholders -
    num_computed_tokens`` formula and is reserved for in-flight *decode*
    tokens. Bumping it for prefill chunks would cause the formula to
    re-schedule the full prompt length on every run-ahead step.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config
        self.chunk_size = self.scheduler_config.max_num_batched_tokens

        # We want to keep track of requests for which the prefill is ongoing.
        # Theoretically, only one request can be prefilled at a time, but we
        # keep a list to be able to batch prefills in the future.
        self.ongoing_prefills: list[Request] = []

        # Per-request counter of prefill-chunk tokens whose optimistic
        # advance has been applied to ``Request.num_computed_tokens`` but
        # not yet committed by ``update_from_output``. The committed prefill
        # position for ``req`` is
        # ``req.num_computed_tokens - self._inflight_prefill_tokens.get(rid, 0)``.
        self._inflight_prefill_tokens: dict[str, int] = {}

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

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Track in-flight prefill-chunk advances per request.

        The base ``Scheduler._update_after_schedule`` optimistically advances
        ``request.num_computed_tokens`` by ``num_scheduled_tokens`` so the next
        ``schedule()`` call (possibly running ahead under async scheduling) can
        pick the next chunk without waiting for the model output. We record
        the per-chunk advance in ``_inflight_prefill_tokens`` so
        ``update_from_output`` can later reconcile the optimistic value
        against what actually committed (after left-padding / prefix-cache
        adjustments).

        We use a private dict instead of upstream's ``num_output_placeholders``
        because the latter participates in the ``num_new_tokens`` formula
        (decode-side semantics). Decode-side placeholders are managed by
        ``AsyncScheduler`` (which our async variant inherits via the MRO);
        here we only handle prefill chunks. Decode requests fall through
        unchanged.
        """
        super()._update_after_schedule(scheduler_output)
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            # ``is_prefill_chunk`` is set by the base method and is True iff
            # the optimistic ``num_computed_tokens`` is still below the prompt
            # length (i.e. we just advanced through a non-final prefill chunk).
            if request.is_prefill_chunk:
                self._inflight_prefill_tokens[req_id] = (
                    self._inflight_prefill_tokens.get(req_id, 0) + n
                )

    def update_from_output(self, scheduler_output, model_runner_output):
        # Update tkv before any early returns
        if isinstance(model_runner_output, SpyreModelRunnerOutput):
            self.tkv = model_runner_output.tkv

        # Handle empty output (async run-ahead). The optimistic advance is
        # rolled back below for the executed-req branch; for the empty branch
        # we do not touch placeholders because no request was executed.
        if not model_runner_output.req_ids:
            if scheduler_output.num_scheduled_tokens:
                import dataclasses

                # Create empty scheduler output to match the empty model output
                empty_scheduler_output = dataclasses.replace(
                    scheduler_output,
                    num_scheduled_tokens={},
                    total_num_scheduled_tokens=0,
                )
                return super().update_from_output(empty_scheduler_output, model_runner_output)
            return super().update_from_output(scheduler_output, model_runner_output)

        assert isinstance(model_runner_output, SpyreModelRunnerOutput), (
            "Expecting an instance of SpyreModelRunnerOutput when doing chunked prefill."
        )

        executed_req_ids = set(model_runner_output.req_ids)

        # Reconcile the optimistic prefill advance: for each executed prefill
        # chunk, compute how many tokens *actually* committed (accounting for
        # left-padding and prefix-cache hits), correct ``num_computed_tokens``
        # by the delta, and decrement the per-request in-flight counter
        # bumped in ``_update_after_schedule``. The per-chunk truth is
        # derivable from the runner output and
        # ``scheduler_output.num_scheduled_tokens``.
        for req_id in executed_req_ids:
            req = self.requests.get(req_id)
            if req is None:
                continue
            scheduled_n = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            inflight = self._inflight_prefill_tokens.get(req_id, 0)
            if scheduled_n == 0 or inflight == 0:
                # Not a prefill-chunk we tracked (decode requests are not
                # recorded in _inflight_prefill_tokens).
                continue

            # Committed prefill position before THIS chunk's optimistic
            # advance. ``inflight`` is the cumulative pending across all
            # in-flight chunks for this request, so subtracting it from the
            # current (optimistic, possibly run-ahead) ``num_computed_tokens``
            # yields the truly committed position prior to this chunk.
            committed_before = req.num_computed_tokens - inflight
            # Optimistic position after THIS chunk would commit (i.e. excluding
            # any further speculatively-scheduled chunks). This matches the
            # semantics of the old snapshot mechanism's ``num_after_batch``.
            optimistic_after_this_chunk = committed_before + scheduled_n
            is_first_chunk = committed_before == 0
            is_last_chunk = optimistic_after_this_chunk >= req.num_prompt_tokens

            if is_first_chunk and not is_last_chunk:
                left_padding = model_runner_output.left_padding.get(req_id, 0)
                prefix_cache_len = model_runner_output.prefix_cache_hit_len.get(req_id, 0)
                adjusted = self.adjust_computed_tokens(
                    computed_tokens=optimistic_after_this_chunk,
                    left_padding=left_padding,
                    prefix_cache_len=prefix_cache_len,
                )
                # Apply only the delta so any speculative run-ahead advance
                # already reflected in ``num_computed_tokens`` is preserved.
                req.num_computed_tokens += adjusted - optimistic_after_this_chunk

            # Decrement (or pop) the in-flight counter for this committed chunk.
            new_inflight = inflight - scheduled_n
            if new_inflight <= 0:
                self._inflight_prefill_tokens.pop(req_id, None)
            else:
                self._inflight_prefill_tokens[req_id] = new_inflight

        # Remove completed prefills (use committed position to avoid keeping a
        # request whose prefill is in flight but already optimistically done).
        self.ongoing_prefills = [
            req for req in self.ongoing_prefills
            if (
                req.num_computed_tokens
                - self._inflight_prefill_tokens.get(req.request_id, 0)
            ) < req.num_prompt_tokens
        ]

        # With async scheduling, scheduler_output may contain requests that
        # weren't executed yet.  Filter to only pass executed requests to the
        # base scheduler.
        filtered_num_scheduled_tokens = {
            req_id: num_tokens
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items()
            if req_id in executed_req_ids
        }
        if len(filtered_num_scheduled_tokens) != len(scheduler_output.num_scheduled_tokens):
            import dataclasses

            filtered_scheduler_output = dataclasses.replace(
                scheduler_output,
                num_scheduled_tokens=filtered_num_scheduled_tokens,
                total_num_scheduled_tokens=sum(filtered_num_scheduled_tokens.values()),
            )
            return super().update_from_output(filtered_scheduler_output, model_runner_output)

        return super().update_from_output(scheduler_output, model_runner_output)

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

    def schedule(self) -> "SchedulerOutput":
        """Apply Spyre chunked-prefill constraints then delegate to the base
        scheduler.

        Spyre hardware constraint: Only one request can perform chunked prefill
        at a time, and a new prefill cannot be mixed with running decodes.

        Enforced by limiting the waiting queue to 1 new request and hiding
        running decode requests from the base scheduler before delegation.
        This pre-filter approach applies to both sync and async modes.
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

        # Re-derive ongoing_prefills from the current running set. Under async
        # run-ahead, ``num_computed_tokens`` is optimistic while
        # ``_inflight_prefill_tokens`` records the in-flight portion; a request
        # is still prefilling iff its committed position is below the prompt
        # length.
        self.ongoing_prefills = [
            req
            for req in self.running
            if (
                req.num_computed_tokens
                - self._inflight_prefill_tokens.get(req.request_id, 0)
            ) < req.num_prompt_tokens
        ]

        # Schedule new requests (one prefill at a time in sync mode)
        num_added_to_waiting = 0
        unscheduled_requests: deque[Request] = deque()

        while holdback_queue:
            request = holdback_queue.popleft()
            if self.can_schedule_prefill(request):
                should_add_to_waiting = len(self.ongoing_prefills) > 0 or num_added_to_waiting == 0

                if should_add_to_waiting:
                    logger.debug(
                        "Scheduling a new request (%d prompt tokens), holding back %d requests",
                        request.num_prompt_tokens,
                        len(holdback_queue) + len(unscheduled_requests),
                    )
                    self.waiting.append(request)
                    num_added_to_waiting += 1
                else:
                    # Sync mode: defer this request
                    unscheduled_requests.append(request)
            else:
                # Can't schedule, restore and stop
                holdback_queue.appendleft(request)
                break

        # Restore unscheduled requests to holdback
        while unscheduled_requests:
            holdback_queue.appendleft(unscheduled_requests.pop())

        assert len(self.ongoing_prefills) <= 1, (
            "Only one request can be prefilled at a time, but got %d" % len(self.ongoing_prefills)
        )
        assert len(self.waiting) == 0 or len(self.ongoing_prefills) == 0, (
            "Cannot schedule new requests while another request prefill is ongoing."
        )
        assert all(r in self.running for r in self.ongoing_prefills), (
            "Ongoing prefill requests must be in the running queue."
        )

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
                # Track in ongoing_prefills so subsequent constraint checks
                # treat these requests as prefilling.
                self.ongoing_prefills.extend(self.waiting)
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

        # delegate to the base scheduler.  In the async variant this resolves
        # to AsyncScheduler.schedule() via the MRO. Reconciliation of the
        # optimistic num_computed_tokens advance happens in
        # update_from_output() via num_output_placeholders rather than via a
        # snapshot deque maintained here.
        outputs = super().schedule()

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
        outputs._spyre_grammar_output = self.get_grammar_bitmask(outputs)
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
        max_context_len = self.model_config.max_model_len

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

        # check that the number of requested tokens can be served for the
        # new sequence (optimal condition)
        # note that the -1 comes from the token we generate during prefill
        cond2 = request.max_tokens - 1 <= (max_context_len - new_req_tkv)
        # check cond2 for all other sequences in the current decode batch
        for req in decoding_requests:
            # current tkv of the (left aligned) decode sequence
            dec_req_tkv = n_blocks * self.block_size + req.num_computed_tokens % self.block_size
            n_generated_output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            max_tokens_remaining = req.max_tokens - n_generated_output_tokens
            # note that the -1 comes from the token we generate during prefill
            cond2_current = max_tokens_remaining - 1 <= (max_context_len - dec_req_tkv)
            cond2 = cond2 and cond2_current
            # early exiting loop if violated 2nd condition
            if not cond2:
                return False

        # check that batch size x tkv is smaller than the max supported number
        # Note: using max_tkv is a conservative upper bound here. For the
        # optimal check we need model runner to return per sequence tkvs
        cond3 = lambda: self.check_batch_tkv_limit_cp(
            request=request,
            new_req_tkv=new_req_tkv,
            running=decoding_requests,
        )

        return cond1 and cond2 and cond3()

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
        """Handles removing finished requests from ongoing_prefills."""
        if isinstance(request_ids, str):
            request_ids = (request_ids,)

        # first defer to vLLM scheduler
        # validates the input requests and generates the output
        aborted_requests = super().finish_requests(
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
        """
        base_stats = super().make_stats(*args, **kwargs)

        if base_stats is not None and base_stats.prefix_cache_stats is not None:
            base_stats.prefix_cache_stats.hits = self.adjust_hit(
                base_stats.prefix_cache_stats.queries, base_stats.prefix_cache_stats.hits
            )

        return base_stats


__all__ = [
    "PoolingSpyreScheduler",
    "ChunkedPrefillSpyreScheduler",
]
