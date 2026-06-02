import torch
from llm_cache import patch_environment
from llm_cache_util import force_engine_shutdown
from spyre_util import ModelInfo
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality
from sendnn_inference.v1.sample.spyre_logits_processor import SpyreBatchUpdate


def test_custom_logits_processor(
    model: ModelInfo, backend, monkeypatch, max_num_seqs, max_model_len, mode: str
):
    """
    Simple test to check if custom logits processors are being registered
    """

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    has_invoked_logits_processor = False

    class DummyLogitsProcessor(LogitsProcessor):
        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            # Required to register LogitsProcessor
            pass

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            # Required to register LogitsProcessor
            pass

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
        logits_processors=[DummyLogitsProcessor],
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

    class SpyLogitsProcessor(LogitsProcessor):
        """
        This logits processor collect the tokens
        """

        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            self.req_info: dict[int, SamplingParams] = {}
            # Saved state for paused requests, keyed by req_id.
            self._paused_info: dict[str, SamplingParams] = {}

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            if not batch_update:
                return

            for index, params, _, _ in batch_update.added:
                self.req_info[index] = params
                nonlocal spy_outputs
                # Use setdefault so a resume-triggered added event does not
                # reset tokens already collected before the pause.
                spy_outputs.setdefault(params.max_tokens, [])

            if SpyreBatchUpdate is not None:
                for dense_index, req_id in getattr(batch_update, "resumed", []):
                    if req_id in self._paused_info:
                        self.req_info[dense_index] = self._paused_info.pop(req_id)

            if SpyreBatchUpdate is not None:
                for dense_index, req_id in getattr(batch_update, "paused", []):
                    if dense_index in self.req_info:
                        self._paused_info[req_id] = self.req_info.pop(dense_index)

            if self.req_info:
                # Process removed requests.
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # Process moved requests, unidirectional move (a->b) and swap
                # (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return
            batch_size = logits.shape[0]
            nonlocal spy_outputs
            for i in range(batch_size):
                params = self.req_info[i]
                token_id = logits[i].argmax(-1).reshape(-1).item()
                spy_outputs[params.max_tokens].append(token_id)
            return logits

    patch_environment(
        backend=backend,
        monkeypatch=monkeypatch,
    )

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=2,
        logits_processors=[SpyLogitsProcessor],
        max_num_batched_tokens=128,
        enable_prefix_caching=mode == "pc",
    )
    prompt = ["1 2 3 4 5 6 7 8 9 " * 10] * 3
    params0 = SamplingParams(max_tokens=5, temperature=0, logprobs=0, ignore_eos=True)
    params1 = SamplingParams(max_tokens=10, temperature=0, logprobs=0, ignore_eos=True)
    params2 = SamplingParams(max_tokens=7, temperature=0, logprobs=0, ignore_eos=True)

    # clear from the warmup
    spy_outputs = {}
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

    This test simulates various scheduler scenarios where requests can be:
    1. Added and finished in the same step
    2. Paused and resumed
    3. Resumed while others finish
    4. Multiple operations happening simultaneously
    """
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
    from vllm.v1.request import Request
    from tests.v1.worker.mock_model import InstrumentedModelRunner

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Track all state transitions and verify correctness
    state_log: list[dict] = []

    class StateTrackingLogitsProcessor(LogitsProcessor):
        """
        Tracks the complete state of the batch at each update.
        Verifies that indices are correct and no overwrites occur.
        """

        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            # Map dense_index -> (req_id, token_count)
            self.active_requests: dict[int, tuple[str, int]] = {}
            # Map req_id -> (last_dense_index, token_count) for paused requests
            self.paused_requests: dict[str, tuple[int, int]] = {}
            self.step_count = 0

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            if not batch_update:
                return

            self.step_count += 1
            step_info = {
                "step": self.step_count,
                "batch_size": batch_update.batch_size,
                "operations": [],
                "active_before": dict(self.active_requests),
                "paused_before": dict(self.paused_requests),
            }

            # Process added requests
            for dense_index, params, prompt_toks, output_toks in batch_update.added:
                req_id = f"req_{params.max_tokens}"  # Use max_tokens as identifier
                token_count = len(output_toks)

                # Verify no overwrite
                if dense_index in self.active_requests:
                    old_req = self.active_requests[dense_index]
                    step_info["operations"].append(
                        {
                            "type": "ERROR_OVERWRITE",
                            "index": dense_index,
                            "old_req": old_req,
                            "new_req": (req_id, token_count),
                        }
                    )

                self.active_requests[dense_index] = (req_id, token_count)
                step_info["operations"].append(
                    {"type": "added", "index": dense_index, "req_id": req_id, "tokens": token_count}
                )

            # Process resumed requests (restore from pause)
            if hasattr(batch_update, "resumed"):
                for dense_index, req_id in batch_update.resumed:
                    if req_id in self.paused_requests:
                        old_index, token_count = self.paused_requests.pop(req_id)
                        self.active_requests[dense_index] = (req_id, token_count)
                        step_info["operations"].append(
                            {
                                "type": "resumed",
                                "index": dense_index,
                                "req_id": req_id,
                                "old_index": old_index,
                                "tokens": token_count,
                            }
                        )

            # Process paused requests (save state)
            if hasattr(batch_update, "paused"):
                for dense_index, req_id in batch_update.paused:
                    if dense_index in self.active_requests:
                        req_info = self.active_requests.pop(dense_index)
                        self.paused_requests[req_id] = (dense_index, req_info[1])
                        step_info["operations"].append(
                            {
                                "type": "paused",
                                "index": dense_index,
                                "req_id": req_id,
                                "tokens": req_info[1],
                            }
                        )

            # Process removed requests
            for dense_index in batch_update.removed:
                if dense_index in self.active_requests:
                    req_info = self.active_requests.pop(dense_index)
                    step_info["operations"].append(
                        {
                            "type": "removed",
                            "index": dense_index,
                            "req_id": req_info[0],
                            "tokens": req_info[1],
                        }
                    )

            # Process moved requests (always swaps like LogitProcessorWrapper)
            for src_idx, dst_idx, _ in batch_update.moved:
                src_req = self.active_requests.get(src_idx)
                dst_req = self.active_requests.get(dst_idx)

                # Always swap both positions (matching LogitProcessorWrapper behavior)
                if src_req:
                    self.active_requests[dst_idx] = src_req
                else:
                    self.active_requests.pop(dst_idx, None)
                if dst_req:
                    self.active_requests[src_idx] = dst_req
                else:
                    self.active_requests.pop(src_idx, None)

                step_info["operations"].append(
                    {
                        "type": "moved",
                        "src": src_idx,
                        "dst": dst_idx,
                        "src_req": src_req,
                        "dst_req": dst_req,
                    }
                )

            step_info["active_after"] = dict(self.active_requests)
            step_info["paused_after"] = dict(self.paused_requests)

            # Verify indices are consecutive immediately after each update
            if self.active_requests:
                indices = sorted(self.active_requests.keys())
                expected = list(range(len(indices)))
                if indices != expected:
                    step_info["operations"].append(
                        {
                            "type": "ERROR_NON_CONSECUTIVE",
                            "actual_indices": indices,
                            "expected_indices": expected,
                            "message": (
                                f"Indices are not consecutive: {indices}, expected {expected}"
                            ),
                        }
                    )

            nonlocal state_log
            state_log.append(step_info)

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            # Increment token count for all active requests
            for idx in self.active_requests:
                req_id, token_count = self.active_requests[idx]
                self.active_requests[idx] = (req_id, token_count + 1)
            return logits

    patch_environment(
        backend=backend,
        monkeypatch=monkeypatch,
    )

    # Build the model runner with our tracking processor
    runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        enable_prefix_caching=mode == "pc",
        model_name=model.name,
        max_num_seqs=4,  # Allow up to 4 concurrent requests
        max_model_len=max_model_len,
        max_num_batched_tokens=128,
    )

    # Replace logits processors with our tracking one
    from sendnn_inference.v1.sample.spyre_logits_processor import build_logitsprocs_for_cb

    runner.input_batch.logitsprocs = build_logitsprocs_for_cb(
        vllm_config=runner.vllm_config,
        device=runner.device,
        is_pin_memory=runner.pin_memory,
        is_pooling_model=False,
        batch_size=4,
        custom_logitsprocs=[StateTrackingLogitsProcessor],
    )

    # Helper to create requests
    def make_request(req_id: str, prompt_len: int, max_tokens: int) -> Request:
        return Request(
            request_id=req_id,
            prompt_token_ids=[42] * prompt_len,
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
            pooling_params=None,
        )

    # Helper to create new request data
    def make_new_req_data(req_id: str, prompt_len: int, max_tokens: int) -> NewRequestData:
        req = make_request(req_id, prompt_len, max_tokens)
        block_ids = list(range(1, (prompt_len + 63) // 64 + 1))
        return NewRequestData.from_request(req, block_ids=(block_ids,))

    # Helper to create cached request data
    def make_cached_req_data(req_states: dict[str, tuple[int, list[int]]]) -> CachedRequestData:
        cached = CachedRequestData.make_empty()
        cached.req_ids = list(req_states.keys())
        cached.num_computed_tokens = [state[0] for state in req_states.values()]
        cached.new_block_ids = [None for _ in req_states]  # Simplified for test
        return cached

    # Add first request: req 0
    req0 = make_new_req_data("req0", 50, 5)
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[req0],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req0": 50},
        total_num_scheduled_tokens=50,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Decode request 0
    cached = make_cached_req_data({"req0": (50, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req0": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Add second request (long one, needs three chunks)
    # Chunked-prefill 1/3 of request 1
    req1 = make_new_req_data("req1", 266, 10)
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[req1],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req1": 128},
        total_num_scheduled_tokens=128,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Decode request 0
    cached = make_cached_req_data({"req0": (51, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req0": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Chunked-prefill 2/3 of request 1
    cached = make_cached_req_data({"req1": (128, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req1": 128},
        total_num_scheduled_tokens=128,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Decode request 0
    cached = make_cached_req_data({"req0": (52, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req0": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Chunked-prefill 3/3 of request 1
    cached = make_cached_req_data({"req1": (256, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req1": 138},
        total_num_scheduled_tokens=138,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Decode requests 0 and 1
    cached = make_cached_req_data({"req0": (53, []), "req1": (266, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req0": 1, "req1": 1},
        total_num_scheduled_tokens=2,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Decode request 0, pause request 1
    cached = make_cached_req_data({"req0": (54, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req0": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Now finish req1, pause req0, and add req2
    req2 = make_new_req_data("req2", 50, 7)
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[req2],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req2": 50},
        total_num_scheduled_tokens=50,
        finished_req_ids={"req1"},
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Verify no overwrites occurred
    for step in state_log:
        for op in step["operations"]:
            assert op["type"] != "ERROR_OVERWRITE", (
                f"Overwrite detected at step {step['step']}: {op}"
            )

    # Clean up: finish remaining requests
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        finished_req_ids={"req1", "req2"},
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Start fresh with req3
    req3 = make_new_req_data("req3", 50, 8)
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[req3],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req3": 50},
        total_num_scheduled_tokens=50,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Decode request 3
    cached = make_cached_req_data({"req3": (50, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req3": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Prefill request 4
    req4 = make_new_req_data("req4", 50, 12)
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[req4],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req4": 50},
        total_num_scheduled_tokens=50,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Simulate pause of req3 (scheduler removes it temporarily)
    # In real scenario, scheduler would not include req3 in cached_reqs
    cached = make_cached_req_data({"req4": (50, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req4": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Resume req3 and finish req4 simultaneously
    cached = make_cached_req_data({"req3": (51, [])})
    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req3": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids={"req4"},
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        free_encoder_mm_hashes=[],
    )
    runner.execute_model(sched_out)

    # Final verification
    print("\n=== State Log Summary ===")
    for step in state_log:
        print(f"\nStep {step['step']}:")
        print(f"  Batch size: {step['batch_size']}")
        print(f"  Operations: {len(step['operations'])}")
        for op in step["operations"]:
            print(f"    - {op['type']}: {op}")
        print(f"  Active after: {step['active_after']}")
        print(f"  Paused after: {step['paused_after']}")

    # Verify no errors occurred during state transitions
    errors = [
        (step["step"], op)
        for step in state_log
        for op in step["operations"]
        if op["type"].startswith("ERROR")
    ]

    if errors:
        print("\n=== ERRORS DETECTED ===")
        for step_num, error_op in errors:
            print(f"Step {step_num}: {error_op['type']}")
            print(f"  Details: {error_op}")

    assert len(errors) == 0, f"Found {len(errors)} errors in state transitions: {errors}"

    # Additional verification: all steps should have consecutive indices
    for step in state_log:
        if step["active_after"]:
            indices = sorted(step["active_after"].keys())
            expected = list(range(len(indices)))
            assert indices == expected, (
                f"Step {step['step']}: Non-contiguous indices {indices}, expected {expected}"
            )

    print("\n=== Test Passed ===")
    print(f"Total steps: {len(state_log)}")
    print(f"Total operations: {sum(len(s['operations']) for s in state_log)}")
