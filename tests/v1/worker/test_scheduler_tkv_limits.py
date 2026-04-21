"""
Tests for scheduler TKV limit enforcement.

This module tests the chunked prefill scheduler's ability to correctly
enforce the max_batch_tkv_limit constraint.
"""

import pytest
from scheduling_utils import create_request_for_scheduler_test, random_prompt
from v1.worker.mock_model import InstrumentedModelRunner
from spyre_util import REFERENCE_MODELS
from sendnn_inference.platform import SpyrePlatform


@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_scheduler_tkv_limits(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the scheduler correctly enforces the TKV limit constraint.

    The test creates 5 requests with varying prompt lengths and a large
    max_tokens value, then runs the scheduler to completion. With the bug
    present, the scheduler incorrectly accepts a batch configuration that
    violates the TKV limit of 131072.

    Expected behavior (when bug is fixed):
    - Scheduler should hold back the next request in queue
      until it is guaranteed not to violate TKV limits later
    - Test should pass without exceeding hardware constraints

    Current behavior (with bug):
    - Scheduler accepts invalid batch configurations
    - Test will fail with assertion errors
    """
    # Setup: Use the default test model
    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]

    # Build model runner with specific constraints
    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=32,
        max_model_len=32768,
        available_blocks=32768,
    )

    # Configure the TKV limit
    scheduler = model_runner.scheduler
    scheduler.max_batch_tkv_limit = 131072
    SpyrePlatform._max_batch_tkv_limit = 131072
    monkeypatch.setenv("VLLM_DT_MAX_BATCH_TKV_LIMIT", "131072")

    # Define prompt lengths for each request
    # This combination is designed to trigger the TKV limit bug
    prompt_lengths = [940, 412, 969, 949, 11946]
    max_tokens = 16384

    # Create and add all requests to the scheduler
    requests = []
    for request_id, prompt_length in enumerate(prompt_lengths):
        prompt = random_prompt(model=model, seed=0, length=prompt_length)
        request = create_request_for_scheduler_test(
            model=model,
            request_id=request_id,
            add_step=0,
            max_tokens=max_tokens,
            prompt=prompt,
            use_golden_token_injection=False,
            generate_hf_results=False,
        ).request
        requests.append(request)
        scheduler.add_request(request)

    # Run the scheduler loop until all requests complete
    # With the bug present, the scheduler will incorrectly accept a batch
    # configuration that exceeds the TKV limit
    while True:
        sched_output = scheduler.schedule()
        output = model_runner.execute_model(sched_output)
        scheduler.update_from_output(sched_output, output)
        if len(scheduler.running) == 0:
            break
