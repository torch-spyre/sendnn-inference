# SPDX-License-Identifier: Apache-2.0
"""Regression test: a grammar-pending request must not stall in-flight decodes.

vLLM's scheduler ships structured-output requests through a separate queue
(``self.skipped_waiting``) precisely so they can sit there with their FSM
still compiling without holding up the main waiting queue. The Spyre
``ChunkedPrefillSpyreScheduler`` drains both ``waiting`` and
``skipped_waiting`` into its own holdback each iteration to enforce its
one-prefill-at-a-time invariant — so it now owns the responsibility of
treating grammar-pending requests correctly.

If the Spyre scheduler does NOT filter grammar-pending requests after the
holdback pass, then whenever such a request sits in the queue the scheduler
hides the running decodes to switch to prefill mode but ``super().schedule()``
can't actually prefill the request — its FSM isn't ready. The result is
alternating "stalled" and "decode" steps that halve throughput for every
other request behind the structured one.

This test verifies that the unstructured requests in front of a
grammar-pending structured request continue to decode on every scheduler
step until they finish, and that once the grammar is signalled ready the
structured request can then start prefilling and producing tokens.

The test only interacts with the runner's public methods
(``execute_new_request`` / ``execute_running_requests``) and reads per-step
progress via the returned ``ModelRunnerOutput``. The grammar's readiness is
gated on a ``threading.Event`` so the test can deterministically drive both
phases — pending grammar, then signalled.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import Future

import pytest
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output.request import StructuredOutputRequest

from scheduling_utils import create_request_for_scheduler_test, random_prompt
from spyre_util import REFERENCE_MODELS
from v1.worker.mock_model import InstrumentedModelRunner

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.worker,
    pytest.mark.skip_global_cleanup,
]


class _PermissiveStubGrammar:
    """Minimal ``StructuredOutputGrammar`` stub: accepts any tokens, never
    terminates. The scheduler advances the FSM by calling ``accept_tokens``
    each step a structured request samples; this stub keeps the scheduler
    path well-formed without our having to compile a real grammar."""

    def accept_tokens(self, request_id: str, tokens) -> bool:
        return True

    def rollback(self, num_tokens: int) -> None:
        return None

    def fill_bitmask(self, bitmask, batch_index: int) -> None:
        return None

    def is_terminated(self) -> bool:
        return False

    def reset(self) -> None:
        return None


def _event_gated_grammar_future(ready: threading.Event) -> Future:
    """Build a ``Future`` that resolves to a ``_PermissiveStubGrammar`` once
    ``ready`` is set. While the event is unset, ``Future.result(timeout=...)``
    raises ``TimeoutError`` — which is what ``StructuredOutputRequest.
    is_grammar_ready`` observes when it polls the future, so the scheduler
    sees the request as "grammar still building."""
    fut: Future = Future()
    fut.set_running_or_notify_cancel()

    def waiter() -> None:
        ready.wait()
        if not fut.done():
            fut.set_result(_PermissiveStubGrammar())

    threading.Thread(target=waiter, daemon=True).start()
    return fut


def _attach_pending_grammar(request, runner, tokenizer, ready_event):
    """Equip a Request with a structured-output request whose grammar Future
    won't resolve until ``ready_event`` is set.

    ``grammar_init`` is the public API that the engine input thread would
    normally call to kick off grammar compilation; calling it here serves
    two purposes: it initializes the structured-output backend so the
    scheduler can later allocate bitmask tensors, and it gives us a real
    Future on the request that we can then swap out for our event-gated
    one. Whatever grammar gets compiled in the background by the original
    ``grammar_init`` call is harmless — we never read it."""
    sampling_params = request.sampling_params
    assert sampling_params is not None
    sampling_params.structured_outputs = StructuredOutputsParams(regex="^b+$")
    request.structured_output_request = StructuredOutputRequest.from_sampling_params(
        sampling_params
    )
    sampling_params._validate_structured_outputs(
        runner.vllm_config.structured_outputs_config, tokenizer
    )

    runner.scheduler.structured_output_manager.grammar_init(request)
    # Replace the real (eventually-completing) Future with our event-gated
    # one. The original Future from grammar_init continues to run in the
    # background pool; its result is dropped when we overwrite the slot.
    request.structured_output_request._grammar = _event_gated_grammar_future(ready_event)

    # Request.__init__ set status to WAITING because there was no
    # structured_outputs on sampling_params at construction time. Match what
    # the engine input thread would have produced.
    request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


def _accumulate(
    out: ModelRunnerOutput,
    output_tokens: dict[str, list[int]],
) -> None:
    for rid, sampled in zip(out.req_ids, out.sampled_token_ids):
        output_tokens[rid].extend(sampled)


def test_grammar_pending_request_does_not_stall_running_decodes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two unstructured requests are pre-filled and start decoding. A third
    structured request is added with its grammar Future unresolved. Phase 1:
    drive steps until the runner stops producing tokens — the unstructured
    requests must finish without any scheduler step being wasted while
    they were still decoding. Phase 2: signal the grammar event and verify
    the structured request can then prefill and produce its own tokens."""

    runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        available_blocks=100,
    )

    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    tokenizer = get_tokenizer(tokenizer_name=model.name, revision=model.revision)

    unstructured_max_tokens = 8
    unstructured = [
        create_request_for_scheduler_test(
            model=model,
            request_id=i,
            add_step=0,
            max_tokens=unstructured_max_tokens,
            prompt=random_prompt(model=model, seed=i, length=8),
            use_golden_token_injection=False,
            generate_hf_results=False,
        )
        for i in range(2)
    ]

    grammar_ready = threading.Event()
    structured = create_request_for_scheduler_test(
        model=model,
        request_id=2,
        add_step=0,
        max_tokens=4,
        prompt=random_prompt(model=model, seed=100, length=8),
        use_golden_token_injection=False,
        generate_hf_results=False,
    )
    _attach_pending_grammar(structured.request, runner, tokenizer, grammar_ready)

    output_tokens: dict[str, list[int]] = defaultdict(list)
    unstructured_ids = {req.request.request_id for req in unstructured}
    structured_id = structured.request.request_id

    # Submit all three requests through the runner. The runner schedules and
    # executes one step per call; the scheduler will interleave prefills and
    # decodes as it sees fit. The structured request enters
    # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR and should not be prefilled.
    for req in unstructured:
        _accumulate(runner.execute_new_request(request=req.request), output_tokens)
    _accumulate(runner.execute_new_request(request=structured.request), output_tokens)

    # ---- PHASE 1: drive until the runner is exhausted ----
    # "Exhausted" = three consecutive scheduler steps produced no work. A
    # healthy scheduler stops producing once both unstructured requests
    # finish; a stalled scheduler alternates between empty and decode steps
    # until the unstructured requests eventually finish anyway.
    #
    # During the loop, track every step where the runner produced nothing
    # while at least one unstructured request was still alive (started but
    # not yet at its token budget). Those are the stalls.
    stalled_steps: list[int] = []
    empty_streak = 0
    step_limit = 100  # safety cap; on a healthy scheduler we exit far earlier
    for step in range(step_limit):
        out = runner.execute_running_requests()

        alive_unstructured = {
            rid
            for rid in unstructured_ids
            if 0 < len(output_tokens[rid]) < unstructured_max_tokens
        }

        if not out.req_ids:
            empty_streak += 1
            if alive_unstructured:
                stalled_steps.append(step)
        else:
            empty_streak = 0

        _accumulate(out, output_tokens)

        if empty_streak >= 3:
            break

    # Every unstructured request should have produced all its tokens.
    for rid in unstructured_ids:
        assert len(output_tokens[rid]) >= unstructured_max_tokens, (
            f"unstructured request {rid} only produced "
            f"{len(output_tokens[rid])}/{unstructured_max_tokens} tokens before "
            f"the runner went idle — a grammar-pending request appears to be "
            f"head-of-line blocking it."
        )

    # And no step should have stalled while one was still alive.
    assert not stalled_steps, (
        f"Scheduler stalled on {len(stalled_steps)} step(s) ({stalled_steps}) "
        f"while an unstructured request was still decoding. A grammar-pending "
        f"request in the queue is head-of-line blocking running decodes — the "
        f"scheduler hid the running decodes to switch to prefill mode, but the "
        f"structured request's FSM wasn't ready so no work landed."
    )

    # The structured request must not have advanced yet — grammar still pending.
    assert len(output_tokens[structured_id]) == 0, (
        f"structured request advanced ({len(output_tokens[structured_id])} "
        f"tokens) before its grammar became ready"
    )

    # ---- PHASE 2: signal grammar ready; structured request must progress ----
    grammar_ready.set()

    for _ in range(20):
        _accumulate(runner.execute_running_requests(), output_tokens)
        if output_tokens[structured_id]:
            break

    assert output_tokens[structured_id], (
        "structured request still didn't produce any tokens after the grammar "
        "event was set"
    )