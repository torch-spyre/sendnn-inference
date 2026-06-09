"""Unit tests for scheduler handling of structured outputs.

Tests the structured output support in sendnn_inference/v1/core/scheduler.py that
preserves structured_output_request on Request objects and attaches grammar
output via _spyre_grammar_output attribute in the chunked prefill scheduler.

These unit tests mock the scheduler dependencies and call the actual schedule() method.
"""

import pytest
from unittest.mock import Mock, patch
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.output import CachedRequestData
from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler
from scheduling_utils import create_request_for_scheduler_test, random_prompt

from v1.worker.mock_model import InstrumentedModelRunner
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.tokenizers import get_tokenizer
from spyre_util import REFERENCE_MODELS

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def mocked_scheduler():
    """Create a mock scheduler with minimal dependencies."""
    # Create a mock vllm_config
    mock_vllm_config = Mock()
    mock_vllm_config.model_config.max_model_len = 2048
    mock_vllm_config.scheduler_config.max_num_batched_tokens = 128
    mock_vllm_config.scheduler_config.max_num_seqs = 4

    # Create scheduler instance with mocked dependencies
    with patch.object(ChunkedPrefillSpyreScheduler, "__init__", lambda x, *args, **kwargs: None):
        scheduler = ChunkedPrefillSpyreScheduler()

    # Set required attributes
    scheduler.vllm_config = mock_vllm_config
    scheduler.model_config = mock_vllm_config.model_config
    scheduler.scheduler_config = mock_vllm_config.scheduler_config
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler.ongoing_prefills = []
    scheduler.chunk_size = 128
    scheduler.do_interleaving = False
    scheduler.previous_step_was_prefill = False
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

    # Mock the base scheduler's schedule method and can_schedule_prefill,
    # but ChunkedPrefillSpyreScheduler.schedule uses the code implementation
    mock_output = Mock()
    mock_output.has_structured_output_requests = False
    mock_output.num_scheduled_tokens = {}
    mock_output.scheduled_new_reqs = []
    mock_output.scheduled_cached_reqs = CachedRequestData.make_empty()
    # mock_output.total_num_scheduled_tokens = 0
    # mock_output.finished_req_ids = set()

    with (
        patch.object(ChunkedPrefillSpyreScheduler, "can_schedule_prefill", return_value=True),
        patch("vllm.v1.core.sched.scheduler.Scheduler.schedule", return_value=mock_output),
    ):
        yield scheduler


class TestSchedulerStructuredOutputHandling:
    """Test that the scheduler preserves structured_output_request on requests."""

    def test_scheduler_preserves_structured_output_request(self, mocked_scheduler):
        """Test that the scheduler preserves structured_output_request on requests."""

        # Create a request with structured outputs
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )

        request = Request(
            request_id="test_req",
            sampling_params=sampling_params,
            prompt_token_ids=list(range(50)),
            arrival_time=0,
            lora_request=None,
            pooling_params=None,
        )

        # Verify structured_output_request is set
        assert request.structured_output_request is not None
        assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

        # Add request to waiting queue
        mocked_scheduler.waiting.append(request)
        # Call the actual schedule method
        mocked_scheduler.schedule()

        # Verify structured_output_request is preserved
        assert request.structured_output_request is not None

    def test_scheduler_handles_request_without_structured_output(self, mocked_scheduler):
        """Test that requests without structured_output_request are unaffected."""

        # Create a request without structured outputs
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
        )

        request = Request(
            request_id="test_req",
            sampling_params=sampling_params,
            prompt_token_ids=list(range(50)),
            arrival_time=0,
            lora_request=None,
            pooling_params=None,
        )

        # Verify structured_output_request is None
        assert request.structured_output_request is None

        # Add request to waiting queue
        mocked_scheduler.waiting.append(request)
        # Call the actual schedule method
        mocked_scheduler.schedule()

        # Verify request is unchanged
        assert request.structured_output_request is None
        # Status may have changed due to base scheduler, but that's OK

    def test_scheduler_handles_multiple_requests_with_structured_outputs(self, mocked_scheduler):
        """Test that multiple requests with structured outputs are all preserved."""

        # Create multiple requests with structured outputs
        requests = []
        for i in range(3):
            sampling_params = SamplingParams(
                max_tokens=20,
                temperature=0.0,
                structured_outputs=StructuredOutputsParams(json_object=True),
            )

            request = Request(
                request_id=f"test_req_{i}",
                sampling_params=sampling_params,
                prompt_token_ids=list(range(50)),
                arrival_time=i,
                lora_request=None,
                pooling_params=None,
            )
            requests.append(request)
            mocked_scheduler.waiting.append(request)

        # Verify all have structured_output_request set
        for request in requests:
            assert request.structured_output_request is not None
            assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

        # Call the actual schedule method
        mocked_scheduler.schedule()

        # Verify all are preserved
        for request in requests:
            assert request.structured_output_request is not None

    def test_scheduler_preserves_other_request_attributes(self, mocked_scheduler):
        """Test that other request attributes are not affected by scheduling."""

        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.5,
            top_p=0.9,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )

        request = Request(
            request_id="test_req",
            sampling_params=sampling_params,
            prompt_token_ids=list(range(50)),
            arrival_time=1.5,
            lora_request=None,
            pooling_params=None,
        )

        # Store original values
        original_request_id = request.request_id
        original_prompt_tokens = list(request.prompt_token_ids) if request.prompt_token_ids else []
        original_arrival_time = request.arrival_time
        original_sampling_params = request.sampling_params

        # Add request to waiting queue
        mocked_scheduler.waiting.append(request)
        # Call the actual schedule method
        mocked_scheduler.schedule()

        # Verify other attributes are unchanged
        assert request.request_id == original_request_id
        assert request.prompt_token_ids == original_prompt_tokens
        assert request.arrival_time == original_arrival_time
        assert request.sampling_params is original_sampling_params
        # structured_output_request is preserved
        assert request.structured_output_request is not None
        assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


class TestSchedulerSimultaneousRequests:
    """Test that the scheduler handles simultaneous structured and regular requests."""

    def test_simultaneous_structured_and_regular_requests(self, mocked_scheduler):
        """Simulate a mixed batch: some requests use json_object, others don't.
        All structured_output_request values should be preserved correctly."""

        structured_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )
        regular_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
        )

        structured_req_1 = Request(
            request_id="struct_1",
            sampling_params=structured_params,
            prompt_token_ids=list(range(50)),
            arrival_time=0,
            lora_request=None,
            pooling_params=None,
        )
        regular_req = Request(
            request_id="regular_1",
            sampling_params=regular_params,
            prompt_token_ids=list(range(40)),
            arrival_time=1,
            lora_request=None,
            pooling_params=None,
        )
        structured_req_2 = Request(
            request_id="struct_2",
            sampling_params=structured_params,
            prompt_token_ids=list(range(60)),
            arrival_time=2,
            lora_request=None,
            pooling_params=None,
        )

        # Verify initial state
        assert structured_req_1.structured_output_request is not None
        assert regular_req.structured_output_request is None
        assert structured_req_2.structured_output_request is not None

        # Add all three to the waiting queue simultaneously
        mocked_scheduler.waiting.append(structured_req_1)
        mocked_scheduler.waiting.append(regular_req)
        mocked_scheduler.waiting.append(structured_req_2)

        mocked_scheduler.schedule()

        # Structured requests should still have their structured_output_request
        assert structured_req_1.structured_output_request is not None
        assert structured_req_2.structured_output_request is not None
        # Regular request should still have None
        assert regular_req.structured_output_request is None

    def test_simultaneous_structured_requests_all_preserved(self, mocked_scheduler):
        """Multiple structured output requests arriving at the same time
        should all be preserved after scheduling."""

        requests = []
        for i in range(4):
            params = SamplingParams(
                max_tokens=20,
                temperature=0.0,
                structured_outputs=StructuredOutputsParams(json_object=True),
            )
            req = Request(
                request_id=f"concurrent_struct_{i}",
                sampling_params=params,
                prompt_token_ids=list(range(30 + i * 10)),
                arrival_time=i * 0.1,
                lora_request=None,
                pooling_params=None,
            )
            requests.append(req)
            mocked_scheduler.waiting.append(req)

        mocked_scheduler.schedule()

        for req in requests:
            assert req.structured_output_request is not None, (
                f"Request {req.request_id} lost its structured_output_request"
            )

    def test_grammar_output_attached_for_mixed_batch(self, mocked_scheduler):
        """Verify _spyre_grammar_output is attached to scheduler output
        when the batch contains a mix of structured and regular requests."""

        structured_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )
        regular_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
        )

        mocked_scheduler.waiting.append(
            Request(
                request_id="struct_req",
                sampling_params=structured_params,
                prompt_token_ids=list(range(50)),
                arrival_time=0,
                lora_request=None,
                pooling_params=None,
            )
        )
        mocked_scheduler.waiting.append(
            Request(
                request_id="regular_req",
                sampling_params=regular_params,
                prompt_token_ids=list(range(40)),
                arrival_time=1,
                lora_request=None,
                pooling_params=None,
            )
        )

        output = mocked_scheduler.schedule()

        # _spyre_grammar_output should be set on the output
        assert hasattr(output, "_spyre_grammar_output")

    def test_grammar_output_attached_for_all_regular_batch(self, mocked_scheduler):
        """When all requests are regular (no structured output),
        _spyre_grammar_output should still be set (may be None)."""

        regular_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
        )

        for i in range(3):
            mocked_scheduler.waiting.append(
                Request(
                    request_id=f"regular_{i}",
                    sampling_params=regular_params,
                    prompt_token_ids=list(range(50)),
                    arrival_time=i,
                    lora_request=None,
                    pooling_params=None,
                )
            )

        output = mocked_scheduler.schedule()

        # _spyre_grammar_output should still be attached (even if None)
        assert hasattr(output, "_spyre_grammar_output")


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
            pc_model_runner.vllm_config.structured_outputs_config, tokenizer
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
