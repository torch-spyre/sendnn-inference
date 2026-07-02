import torch
from llm_cache import patch_environment
from llm_cache_util import force_engine_shutdown
from logits_processor_utils import (
    DummyLogitsProcessor,
    SpyLogitsProcessor,
    StateTrackingLogitsProcessorWrapper,
    execute_step,
)
from spyre_util import ModelInfo
from vllm import LLM, SamplingParams


def test_custom_logits_processor(
    model: ModelInfo, backend, monkeypatch, max_num_seqs, max_model_len, mode: str
):
    """
    Simple test to check if custom logits processors are being registered
    """

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    has_invoked_logits_processor = False

    class TestDummyLogitsProcessor(DummyLogitsProcessor):
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            nonlocal has_invoked_logits_processor
            has_invoked_logits_processor = True
            return logits

    patch_environment(
        backend=backend,
        monkeypatch=monkeypatch,
    )

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=128,
        enable_prefix_caching=mode == "pc",
        logits_processors=[TestDummyLogitsProcessor],
    )
    prompt = "Hello Logits Processors"
    params = SamplingParams(max_tokens=5, temperature=0, logprobs=0)

    spyre_model.generate(prompt, params)
    force_engine_shutdown(spyre_model)

    assert has_invoked_logits_processor


# TODO: validate that this test case is valid for chunked prefill
def test_logits_processor_cp(model: ModelInfo, backend, monkeypatch, max_model_len, mode: str):
    """
    Test if the state of logits processors are correct due to the switch of
    prefill/decode in a step engine. The LLM is initialized with bs=2,
    we send 3 requests, one of them should be waiting for the other 2
    to complete. The first request should finish and give its slot to
    the last one. The logits processors will do a greedy sampling
    decoding to emulate the 'state' of the logit processor. After
    the generation we assert that the generated output is the same
    for the spy and vllm.
    """

    # Same process to ease things
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Hack to collect outputs from logits, the key
    # is the max_tokens to ease identify the requests
    spy_outputs: dict[int, list[int]] = {}

    class TestSpyLogitsProcessor(SpyLogitsProcessor):
        """Test-specific spy processor that uses the shared spy_outputs dict."""

        def __init__(self, vllm_config, device, is_pin_memory):
            super().__init__(vllm_config, device, is_pin_memory, spy_outputs)

    patch_environment(
        backend=backend,
        monkeypatch=monkeypatch,
    )

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=2,
        logits_processors=[TestSpyLogitsProcessor],
        max_num_batched_tokens=128,
        enable_prefix_caching=mode == "pc",
    )
    prompt = ["Hello Logits Processors"] * 3
    params0 = SamplingParams(max_tokens=5, temperature=0, logprobs=0, ignore_eos=True)
    params1 = SamplingParams(max_tokens=10, temperature=0, logprobs=0, ignore_eos=True)
    params2 = SamplingParams(max_tokens=7, temperature=0, logprobs=0, ignore_eos=True)

    # clear from the warmup
    spy_outputs.clear()
    params = [params0, params1, params2]
    outputs = spyre_model.generate(prompt, params)
    force_engine_shutdown(spyre_model)

    assert spy_outputs[5] == outputs[0].outputs[0].token_ids
    assert spy_outputs[10] == outputs[1].outputs[0].token_ids
    assert spy_outputs[7] == outputs[2].outputs[0].token_ids


def test_logits_processor_advanced(
    model: ModelInfo, backend, monkeypatch, max_model_len, mode: str
):
    """
    Complex test for logits processor state management with controlled SchedulerOutput.

    Tests multiple simultaneous operations:
    - Adding new requests while finishing others
    - Pausing and resuming requests
    - Verifying correct index management
    - Ensuring no state overwrites occur
    - Verifying that the spy logits processor produces correct tokens

    This test simulates various scheduler scenarios where requests can be:
    1. Added and finished in the same step
    2. Paused and resumed
    3. Resumed while others finish
    4. Multiple operations happening simultaneously

    Uses SpyLogitsProcessor as the inner processor to verify token generation.
    """
    from v1.worker.mock_model import InstrumentedModelRunner
    from vllm.v1.sample.logits_processor.state import LogitsProcessors

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Track all state transitions and verify correctness
    state_log: list[dict] = []

    # Track spy outputs to verify token generation
    spy_outputs: dict[int, list[int]] = {}

    # Track actual generated outputs from model runner
    actual_outputs: dict[str, list[int]] = {}

    patch_environment(
        backend=backend,
        monkeypatch=monkeypatch,
    )

    # Build the model runner
    runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        enable_prefix_caching=mode == "pc",
        model_name=model.name,
        max_num_seqs=4,  # Allow up to 4 concurrent requests
        max_model_len=max_model_len,
        max_num_batched_tokens=128,
    )

    # Create a SpyLogitsProcessor factory that uses our spy_outputs dict
    def create_spy_processor(vllm_config, device, is_pin_memory):
        return SpyLogitsProcessor(vllm_config, device, is_pin_memory, spy_outputs)

    # Replace logits processors with our tracking wrapper that wraps SpyLogitsProcessor
    tracking_wrapper = StateTrackingLogitsProcessorWrapper(
        create_spy_processor,
        runner.vllm_config,
        runner.device,
        runner.pin_memory,
        4,
        state_log,
    )
    runner.input_batch.logitsprocs = LogitsProcessors([tracking_wrapper])

    # Get reference to our tracking wrapper
    processor = tracking_wrapper

    # Add first request: req 0
    execute_step(
        runner,
        processor,
        new_reqs=[("req0", 50, 10)],
        num_scheduled_tokens={"req0": 50},
        expected_active={0: ("req0", 1)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Decode request 0
    execute_step(
        runner,
        processor,
        cached_reqs=[("req0", 50)],
        num_scheduled_tokens={"req0": 1},
        expected_active={0: ("req0", 2)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Add request 1 (long one, needs three chunks)
    # Chunked-prefill 1/3 of request 1
    # Chunked-prefill is not added to input_batch unless it is the last chunk
    execute_step(
        runner,
        processor,
        new_reqs=[("req1", 266, 11)],
        num_scheduled_tokens={"req1": 128},
        expected_active={},
        expected_paused={"req0"},
        actual_outputs=actual_outputs,
    )

    # Decode request 0
    execute_step(
        runner,
        processor,
        cached_reqs=[("req0", 51)],
        num_scheduled_tokens={"req0": 1},
        expected_active={0: ("req0", 3)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Chunked-prefill 2/3 of request 1
    execute_step(
        runner,
        processor,
        cached_reqs=[("req1", 128)],
        num_scheduled_tokens={"req1": 128},
        expected_active={},
        expected_paused={"req0"},
        actual_outputs=actual_outputs,
    )

    # Decode request 0
    execute_step(
        runner,
        processor,
        cached_reqs=[("req0", 52)],
        num_scheduled_tokens={"req0": 1},
        expected_active={0: ("req0", 4)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Chunked-prefill 3/3 of request 1
    execute_step(
        runner,
        processor,
        cached_reqs=[("req1", 256)],
        num_scheduled_tokens={"req1": 138},
        expected_active={0: ("req1", 1)},
        expected_paused={"req0"},
        actual_outputs=actual_outputs,
    )

    # Decode requests 0 and 1
    execute_step(
        runner,
        processor,
        cached_reqs=[("req0", 53), ("req1", 266)],
        num_scheduled_tokens={"req0": 1, "req1": 1},
        expected_active={0: ("req1", 2), 1: ("req0", 5)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Decode request 0, pause request 1
    execute_step(
        runner,
        processor,
        cached_reqs=[("req0", 54)],
        num_scheduled_tokens={"req0": 1},
        expected_active={0: ("req0", 6)},
        expected_paused={"req1"},
        actual_outputs=actual_outputs,
    )

    # Finish req1, pause req0, and add req2
    execute_step(
        runner,
        processor,
        new_reqs=[("req2", 50, 12)],
        num_scheduled_tokens={"req2": 50},
        finished_req_ids={"req1"},
        expected_active={0: ("req2", 1)},
        expected_paused={"req0"},
        actual_outputs=actual_outputs,
    )

    # Clean up: finish remaining requests
    execute_step(
        runner,
        processor,
        finished_req_ids={"req0", "req2"},
        expected_active={},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Start fresh with req3
    execute_step(
        runner,
        processor,
        new_reqs=[("req3", 50, 13)],
        num_scheduled_tokens={"req3": 50},
        expected_active={0: ("req3", 1)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Prefill request 4
    execute_step(
        runner,
        processor,
        new_reqs=[("req4", 50, 14)],
        num_scheduled_tokens={"req4": 50},
        expected_active={0: ("req4", 1)},
        expected_paused={"req3"},
        actual_outputs=actual_outputs,
    )

    # Decode request 4, keep request 3 paused
    execute_step(
        runner,
        processor,
        cached_reqs=[("req4", 50)],
        num_scheduled_tokens={"req4": 1},
        expected_active={0: ("req4", 2)},
        expected_paused={"req3"},
        actual_outputs=actual_outputs,
    )

    # Resume req3 and finish req4 simultaneously
    execute_step(
        runner,
        processor,
        cached_reqs=[("req3", 51)],
        num_scheduled_tokens={"req3": 1},
        finished_req_ids={"req4"},
        expected_active={0: ("req3", 2)},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Finish request 3
    execute_step(
        runner,
        processor,
        finished_req_ids={"req3"},
        expected_active={},
        expected_paused=set(),
        actual_outputs=actual_outputs,
    )

    # Verify that actual outputs match spy outputs for each request
    # req0: max_tokens=10, req1: max_tokens=11, req2: max_tokens=12,
    # req3: max_tokens=13, req4: max_tokens=14
    assert "req0" in actual_outputs, "Expected actual_outputs to contain tokens for req0"
    assert len(actual_outputs["req0"]) == 6, (
        f"Expected 6 tokens for req0, got {len(actual_outputs['req0'])}"
    )
    assert actual_outputs["req0"] == spy_outputs[10], (
        f"Token mismatch for req0: {actual_outputs['req0']} != {spy_outputs[10]}"
    )

    assert "req1" in actual_outputs, "Expected actual_outputs to contain tokens for req1"
    assert len(actual_outputs["req1"]) == 2, (
        f"Expected 2 tokens for req1, got {len(actual_outputs['req1'])}"
    )
    assert actual_outputs["req1"] == spy_outputs[11], (
        f"Token mismatch for req1: {actual_outputs['req1']} != {spy_outputs[11]}"
    )

    assert "req2" in actual_outputs, "Expected actual_outputs to contain tokens for req2"
    assert len(actual_outputs["req2"]) == 1, (
        f"Expected 1 tokens for req2, got {len(actual_outputs['req2'])}"
    )
    assert actual_outputs["req2"] == spy_outputs[12], (
        f"Token mismatch for req2: {actual_outputs['req2']} != {spy_outputs[12]}"
    )

    assert "req3" in actual_outputs, "Expected actual_outputs to contain tokens for req3"
    assert len(actual_outputs["req3"]) == 2, (
        f"Expected 2 tokens for req3, got {len(actual_outputs['req3'])}"
    )
    assert actual_outputs["req3"] == spy_outputs[13], (
        f"Token mismatch for req3: {actual_outputs['req3']} != {spy_outputs[13]}"
    )

    assert "req4" in actual_outputs, "Expected actual_outputs to contain tokens for req4"
    assert len(actual_outputs["req4"]) == 2, (
        f"Expected 2 tokens for req4, got {len(actual_outputs['req4'])}"
    )
    assert actual_outputs["req4"] == spy_outputs[14], (
        f"Token mismatch for req4: {actual_outputs['req4']} != {spy_outputs[14]}"
    )
