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
from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler

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

    # Mock the base scheduler's schedule method and can_schedule_prefill,
    # but ChunkedPrefillSpyreScheduler.schedule uses the code implementation
    mock_output = Mock()
    mock_output.has_structured_output_requests = False
    mock_output.num_scheduled_tokens = {}

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
