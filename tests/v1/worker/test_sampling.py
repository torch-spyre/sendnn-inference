"""Tests for sampling separation.

Tests that sampling is separated from model execution to enable
async grammar preparation while the model runs.
"""

import pytest
from vllm import SamplingParams

from v1.worker.mock_model import InstrumentedModelRunner
from sendnn_inference.v1.worker.spyre_model_runner import ChunkedPrefillModelRunner
from spyre_util import REFERENCE_MODELS, create_random_request


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_execute_model_forward_pass_only(monkeypatch: pytest.MonkeyPatch):
    """Test that model runner's execute_model only does forward pass."""

    # Setup: Use the default test model
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]

    # Build model runner
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create a request using spyre_util helper
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
    )

    request = create_random_request(
        request_id=0,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=0,
    )

    scheduler.add_request(request)

    # Schedule
    sched_output = scheduler.schedule()

    # Verify grammar output is attached by scheduler
    assert hasattr(sched_output, "_spyre_grammar_output")

    # Execute model using parent class method - should return tuple (logits, is_prefill, model_input, t0)
    result = ChunkedPrefillModelRunner.execute_model(model_runner, sched_output)
    assert isinstance(result, tuple), "execute_model should return a tuple for sampling"
    assert len(result) == 4, "tuple should have 4 elements"
    logits, is_prefill, model_input, t0 = result
    assert logits is not None, "logits should not be None"
    assert isinstance(is_prefill, bool), "is_prefill should be a bool"
    assert model_input is not None, "model_input should not be None"
    assert isinstance(t0, float), "t0 should be a float"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_sampling_state_storage(monkeypatch: pytest.MonkeyPatch):
    """Test that execute_model stores sampling state."""

    # Setup model runner
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create a request using spyre_util helper
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
    )

    request = create_random_request(
        request_id=1,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=1,
    )

    scheduler.add_request(request)
    sched_output = scheduler.schedule()

    # Execute model using parent class method - should return tuple with logits and metadata
    result = ChunkedPrefillModelRunner.execute_model(model_runner, sched_output)
    assert isinstance(result, tuple), "execute_model should return a tuple"
    assert len(result) == 4, "tuple should have 4 elements"
    logits, is_prefill, model_input, t0 = result
    assert logits is not None, "logits should not be None"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_worker_execute_model_handles_tuple(monkeypatch: pytest.MonkeyPatch):
    """Test that worker's execute_model properly handles tuple from model runner."""

    # Setup model runner
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create a request
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
    )

    request = create_random_request(
        request_id=2,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=2,
    )

    scheduler.add_request(request)
    sched_output = scheduler.schedule()

    # Execute model using mock's execute_model (which handles tuple)
    # This should return ModelRunnerOutput, not tuple
    result = model_runner.execute_model(sched_output)
    assert not isinstance(result, tuple), "Mock's execute_model should return ModelRunnerOutput"
    assert hasattr(result, "req_ids"), "Result should be ModelRunnerOutput with req_ids"
    assert hasattr(result, "sampled_token_ids"), "Result should have sampled_token_ids"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_sampling_with_grammar(monkeypatch: pytest.MonkeyPatch):
    """Test that sampling works correctly with grammar constraints."""
    from vllm.sampling_params import StructuredOutputsParams

    # Setup model runner
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create a request with structured outputs
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )

    request = create_random_request(
        request_id=3,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=3,
    )

    scheduler.add_request(request)
    sched_output = scheduler.schedule()

    # Verify grammar output is attached
    assert hasattr(sched_output, "_spyre_grammar_output")

    # Execute model - mock should handle tuple and apply grammar
    result = model_runner.execute_model(sched_output)
    assert not isinstance(result, tuple), "Mock should return ModelRunnerOutput"
    assert hasattr(result, "sampled_token_ids"), "Result should have sampled tokens"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_multiple_requests_sampling(monkeypatch: pytest.MonkeyPatch):
    """Test sampling separation with multiple requests in batch."""

    # Setup model runner
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create multiple requests
    for i in range(3):
        sampling_params = SamplingParams(
            max_tokens=10,
            temperature=0.0,
            ignore_eos=True,
        )

        request = create_random_request(
            request_id=10 + i,
            num_tokens=50,
            sampling_params=sampling_params,
            from_model_vocab=True,
            model=model,
            seed=10 + i,
        )

        scheduler.add_request(request)

    sched_output = scheduler.schedule()

    # Execute model with multiple requests
    result = model_runner.execute_model(sched_output)
    assert not isinstance(result, tuple), "Mock should return ModelRunnerOutput"
    assert len(result.req_ids) > 0, "Should have processed requests"
    assert result.sampled_token_ids is not None, "Should have sampled tokens"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_prefill_vs_decode_sampling(monkeypatch: pytest.MonkeyPatch):
    """Test that sampling works correctly for both prefill and decode phases."""

    # Setup model runner
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create a request
    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0.0,
        ignore_eos=True,
    )

    request = create_random_request(
        request_id=20,
        num_tokens=50,
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=20,
    )

    scheduler.add_request(request)

    # First step: prefill
    sched_output = scheduler.schedule()
    result = model_runner.execute_model(sched_output)
    scheduler.update_from_output(sched_output, result)

    assert not isinstance(result, tuple), "Should return ModelRunnerOutput"
    assert len(result.req_ids) > 0, "Should have processed request"

    # Second step: decode
    sched_output = scheduler.schedule()
    result = model_runner.execute_model(sched_output)
    scheduler.update_from_output(sched_output, result)

    assert not isinstance(result, tuple), "Should return ModelRunnerOutput for decode"
    assert result.sampled_token_ids is not None, "Should have sampled tokens in decode"


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_incomplete_prefill_no_sampling(monkeypatch: pytest.MonkeyPatch):
    """Test that incomplete prefills don't trigger sampling."""

    # Setup model runner with small chunk size to force multi-chunk prefill
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=128,  # Small chunk size
        max_num_seqs=4,
        max_model_len=2048,
    )

    scheduler = model_runner.scheduler

    # Create a request with long prompt (will need multiple chunks)
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        ignore_eos=True,
    )

    request = create_random_request(
        request_id=30,
        num_tokens=300,  # Long prompt requiring multiple chunks
        sampling_params=sampling_params,
        from_model_vocab=True,
        model=model,
        seed=30,
    )

    scheduler.add_request(request)

    # First chunk: incomplete prefill
    sched_output = scheduler.schedule()
    result = ChunkedPrefillModelRunner.execute_model(model_runner, sched_output)

    # For incomplete prefill, should return prefill_output, not tuple
    assert not isinstance(result, tuple), "Incomplete prefill should not return tuple"
    assert hasattr(result, "req_ids"), "Should return ModelRunnerOutput"


# Made with Bob
