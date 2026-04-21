"""Unit tests for scheduler handling of structured outputs.

Tests the fix in sendnn_inference/v1/core/scheduler.py that strips
structured_output_request from Request objects in the chunked prefill scheduler.

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
    with (
        patch.object(ChunkedPrefillSpyreScheduler, "can_schedule_prefill", return_value=True),
        patch("vllm.v1.core.sched.scheduler.Scheduler.schedule", return_value=Mock()),
    ):
        yield scheduler


class TestSchedulerStructuredOutputHandling:
    """Test that the scheduler strips structured_output_request from requests."""

    def test_scheduler_strips_structured_output_request(self, mocked_scheduler, caplog_sendnn_inference):
        """Test that the scheduler removes structured_output_request from new requests."""

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

        # Verify structured_output_request was stripped
        assert request.structured_output_request is None
        assert request.status == RequestStatus.WAITING

        # Verify warning was logged
        assert any(
            "Removing structured output" in record.message for record in caplog_sendnn_inference.records
        )

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

    def test_scheduler_handles_multiple_requests_with_structured_outputs(
        self, mocked_scheduler, caplog_sendnn_inference
    ):
        """Test that multiple requests with structured outputs are all stripped."""

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

        # Verify all were stripped
        for request in requests:
            assert request.structured_output_request is None
            assert request.status == RequestStatus.WAITING

        # Verify warnings were logged for each request
        warning_count = sum(
            1
            for record in caplog_sendnn_inference.records
            if "Removing structured output" in record.message
        )
        assert warning_count == 3

    def test_scheduler_only_strips_when_can_schedule_prefill_true(self, mocked_scheduler):
        """Test that structured_output_request is only stripped when request can be scheduled."""

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

        # Add request to waiting queue
        mocked_scheduler.waiting.append(request)
        # Mock can_schedule_prefill to return False (request cannot be scheduled)
        with patch.object(ChunkedPrefillSpyreScheduler, "can_schedule_prefill", return_value=False):
            # Call the actual schedule method
            mocked_scheduler.schedule()

        # Verify structured_output_request was NOT stripped (request wasn't scheduled)
        assert request.structured_output_request is not None

    def test_scheduler_preserves_other_request_attributes(
        self, mocked_scheduler, caplog_sendnn_inference
    ):
        """Test that other request attributes are not affected when stripping."""

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
        # But structured_output_request should be None
        assert request.structured_output_request is None
        assert request.status == RequestStatus.WAITING


# Made with Bob
