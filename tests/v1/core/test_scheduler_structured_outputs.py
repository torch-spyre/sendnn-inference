"""Unit tests for scheduler handling of structured outputs.

Tests the structured output support in sendnn_inference/v1/core/scheduler.py
that promotes the grammar status (WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR →
WAITING) when a grammar becomes ready during schedule(), and that requests
with a still-pending grammar remain blocked.

These unit tests mock the scheduler dependencies and call the actual
schedule() method.
"""

import pytest
from unittest.mock import Mock, patch
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.output import SchedulerOutput
from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler
from scheduling_utils import create_request_for_scheduler_test, random_prompt

from v1.worker.mock_model import InstrumentedModelRunner
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.tokenizers import get_tokenizer
from spyre_util import REFERENCE_MODELS

pytestmark = pytest.mark.skip_global_cleanup


def _make_structured_request(request_id: str, arrival_time: float = 0) -> Request:
    """Build a grammar-pending structured-output request."""
    return Request(
        request_id=request_id,
        sampling_params=SamplingParams(
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
        prompt_token_ids=list(range(30)),
        arrival_time=arrival_time,
        lora_request=None,
        pooling_params=None,
    )


def _make_regular_request(request_id: str, arrival_time: float = 0) -> Request:
    """Build a plain (non-structured-output) request."""
    return Request(
        request_id=request_id,
        sampling_params=SamplingParams(max_tokens=20, temperature=0.0),
        prompt_token_ids=list(range(30)),
        arrival_time=arrival_time,
        lora_request=None,
        pooling_params=None,
    )


@pytest.fixture
def mocked_scheduler():
    """Minimal ChunkedPrefillSpyreScheduler with real schedule() but mocked
    infrastructure (kv-cache, base Scheduler.schedule, etc.)."""
    mock_vllm_config = Mock()
    mock_vllm_config.model_config.max_model_len = 2048
    mock_vllm_config.scheduler_config.max_num_batched_tokens = 128
    mock_vllm_config.scheduler_config.max_num_seqs = 4

    with patch.object(ChunkedPrefillSpyreScheduler, "__init__", lambda x, *args, **kwargs: None):
        scheduler = ChunkedPrefillSpyreScheduler()

    scheduler.vllm_config = mock_vllm_config
    scheduler.model_config = mock_vllm_config.model_config
    scheduler.scheduler_config = mock_vllm_config.scheduler_config
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler.ongoing_prefills = []
    scheduler.paused_decoding_requests = []
    scheduler.chunk_size = 128
    scheduler.do_interleaving = False
    scheduler.step_is_prefill = False
    scheduler.max_num_running_reqs = 4
    scheduler.tkv = 0
    scheduler.block_size = 64
    scheduler.n_free_blocks = 100
    scheduler.max_batch_tkv_limit = "8192"
    scheduler.available_blocks = 1
    scheduler.total_reserved_blocks = 0
    scheduler.reserved_blocks = dict[str, int]()
    scheduler._get_required_blocks = lambda x, *args, **kwargs: (0, 0)
    scheduler._get_free_blocks = lambda *args, **kwargs: 1
    scheduler.pause_events = 0
    scheduler.resume_events = 0
    scheduler.long_output_prio = False

    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (None, 0)

    real_output = SchedulerOutput.make_empty()

    with (
        patch.object(ChunkedPrefillSpyreScheduler, "can_schedule_prefill", return_value=True),
        patch("vllm.v1.core.sched.scheduler.Scheduler.schedule", return_value=real_output),
    ):
        yield scheduler


def test_grammar_ready_request_promoted_to_waiting(mocked_scheduler):
    """A structured-output request whose grammar becomes ready between
    schedule() calls must be promoted WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR →
    WAITING so the scheduler can prefill it.

    """
    request = _make_structured_request("struct_req")
    assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert request.structured_output_request.grammar is None

    request.structured_output_request.grammar = Mock()

    mocked_scheduler.waiting.append(request)
    mocked_scheduler.schedule()

    assert request.status == RequestStatus.WAITING, (
        "Request with a ready grammar was not promoted from "
        "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR to WAITING"
    )


def test_mixed_batch_only_ready_grammar_requests_promoted(mocked_scheduler):
    """In a mixed waiting queue only the request whose grammar is ready must
    be promoted; the one still compiling must remain blocked.
    """
    ready_req = _make_structured_request("ready", arrival_time=0)
    pending_req = _make_structured_request("pending", arrival_time=1)
    ready_req.structured_output_request.grammar = Mock()

    mocked_scheduler.waiting.append(ready_req)
    mocked_scheduler.waiting.append(pending_req)
    mocked_scheduler.schedule()

    assert ready_req.status == RequestStatus.WAITING, (
        "ready_req with compiled grammar was not promoted to WAITING"
    )
    assert pending_req.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, (
        "pending_req was promoted despite its grammar not being ready"
    )

def test_grammar_not_ready_request_stays_blocked(mocked_scheduler):
    """A structured-output request whose grammar is still compiling must NOT
    be promoted — it must remain WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR.
    """
    request = _make_structured_request("struct_req_pending")
    assert request.structured_output_request.grammar is None

    mocked_scheduler.waiting.append(request)
    mocked_scheduler.schedule()

    assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, (
        "Request was prematurely promoted while its grammar was still compiling"
    )


def test_regular_request_not_blocked_alongside_pending_grammar(mocked_scheduler):
    """A regular (non-structured-output) request in the same waiting queue as
    a grammar-pending request must be picked up for prefill (step_is_prefill=True).
    """
    structured_req = _make_structured_request("struct", arrival_time=0)
    regular_req = _make_regular_request("regular", arrival_time=1)

    mocked_scheduler.waiting.append(structured_req)
    mocked_scheduler.waiting.append(regular_req)
    mocked_scheduler.schedule()

    # The regular request has no grammar constraint, so the scheduler must
    # have selected it for prefill — step_is_prefill must be True.
    assert mocked_scheduler.step_is_prefill is True, (
        "Scheduler failed to schedule the regular request for prefill; "
        "the ready_to_prefill filter likely excluded it incorrectly"
    )

def test_sparse_index_grammar_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    """In this scenario we schedule two requests with structured outputs. The
    first one will drop out of the batch earlier, making a hole in the sparse
    index. This is to trigger a known bug when the sparse index is not
    contiguous.
    """
    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=64,
        available_blocks=100,
    )
    pc_model_runner.scheduler.structured_output_manager._use_async_grammar_compilation = False

    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    tokenizer = get_tokenizer(tokenizer_name=model.name, revision=model.revision)

    prompt1 = random_prompt(model=model, seed=0, length=64)
    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=3,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=4,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    # Initialize grammars and requests
    for request in [request1, request2]:
        assert (sampling_params := request.request.sampling_params) is not None
        sampling_params.structured_outputs = StructuredOutputsParams(regex=".*")  # accept anything
        request.request.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        sampling_params._validate_structured_outputs(
            model_config=pc_model_runner.vllm_config.model_config,
            structured_outputs_config=pc_model_runner.vllm_config.structured_outputs_config,
            tokenizer=tokenizer,
        )
        pc_model_runner.scheduler.structured_output_manager.grammar_init(request.request)

        assert (structured := request.request.structured_output_request) is not None
        # Wait for grammar to be ready
        while not structured.is_grammar_ready:
            pass

    # Run prefill of request 1
    pc_model_runner.execute_new_request(request=request1.request)
    # Run first decode of request 1
    pc_model_runner.execute_running_requests()

    # Run prefill of request 2
    pc_model_runner.execute_new_request(request=request2.request)

    for i in range(4):
        # Run decode of requests 1 and 2
        pc_model_runner.execute_running_requests()
