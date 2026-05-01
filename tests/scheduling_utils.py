from collections.abc import Iterable
import copy
import dataclasses
import os
from collections import defaultdict, deque
from typing import Any, Callable

import pytest
from llm_cache import get_cached_engine
from output_util import (
    ISCLOSE_ABS_TOL,
    ISCLOSE_ABS_TOL_QUANTIZATION,
    compare_results,
    generate_hf_output,
)
from spyre_util import ModelInfo, create_random_request
from typing_extensions import deprecated
from vllm import SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)


from sendnn_inference.v1.core.scheduler import (
    ChunkedPrefillSpyreScheduler,
)


DISABLE_ASSERTS = False  # used for debugging


def augment_checked_steps(checked_steps: list[dict[str, Any]]) -> deque[dict[str, Any]]:
    # Augment checked_steps: add in-between normal decode steps
    checked_steps = deque(checked_steps)
    all_checked_steps = deque()
    prev_step = None
    for step in range(checked_steps[-1]["step"] + 1):
        if checked_steps and step == checked_steps[0]["step"]:
            prev_step = checked_steps.popleft()
            all_checked_steps.append(prev_step)
        elif prev_step is not None:
            assert prev_step["step"] == step - 1
            new_step = copy.deepcopy(prev_step)
            new_step["step"] = step
            new_step["tkv"] += 1
            all_checked_steps.append(new_step)
            prev_step = new_step
    return all_checked_steps


@dataclasses.dataclass
class SchedulerTestRequest:
    """Little struct for passing around vllm.v1.Requests for scheduler tests.
    The tests often need to know at which step a request is added to the engine.
    """

    add_step: int  # Step that the request will be added to the engine
    request: Request
    hf_output: Any  # hf transformers output for this request


def random_prompt(model: ModelInfo, seed: int, length: int) -> list[int]:
    # Generate a random prompt with valid token ids for this model
    return create_random_request(
        request_id=0,
        model=model,
        from_model_vocab=True,
        sampling_params=SamplingParams.from_optional(),
        seed=seed,
        num_tokens=length,
    ).prompt_token_ids


def create_request_for_scheduler_test(
    model: ModelInfo,
    request_id: int,
    add_step: int,
    max_tokens: int,
    prompt: list[int],
    use_golden_token_injection: bool,
    generate_hf_results: bool = True,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
) -> SchedulerTestRequest:
    # Creates a request out of a prompt, for use with the scheduler tests.
    # Can add golden token injection, which will ensure that the vllm output
    # matches the hf output so that we can do logits comparisons on identical
    # token outputs.

    sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=0.0, logprobs=0, ignore_eos=True
    )

    if generate_hf_results:
        hf_results = generate_hf_output(
            model=model,
            prompts=[prompt],
            max_new_tokens=max_tokens,
            ignore_eos=True,
        )
        hf = hf_results[0]
    else:
        hf = None

    if use_golden_token_injection:
        abs_tol = ISCLOSE_ABS_TOL_QUANTIZATION if model.is_quantized else ISCLOSE_ABS_TOL

        sampling_params.extra_args = {
            "golden_token_injector": {
                "expected_token_ids": hf["token_ids"],
                "expected_logprobs": hf["logprobs"],
                "error_threshold": abs_tol,
                "label": f"#{request_id}",
            }
        }

    if block_hasher is None:
        caching_hash_fn = get_hash_fn_by_name("sha256")
        init_none_hash(caching_hash_fn)
        block_hasher = get_request_block_hasher(64, caching_hash_fn)

    request = Request(
        request_id=str(request_id),
        sampling_params=sampling_params,
        prompt_token_ids=prompt,
        arrival_time=0,
        lora_request=None,
        pooling_params=None,
        cache_salt=None,
        block_hasher=block_hasher,
    )
    return SchedulerTestRequest(add_step=add_step, request=request, hf_output=hf)


def generate_prompts(
    model: ModelInfo,
    steps_add_reqs: list[int],
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    from_model_vocab: bool = False,
    seeds: list[int] = None,
):
    generated_prompts = []

    # Create random requests of specified lengths and max_tokens
    # Need to do before setting up the vLLM engine, otherwise test random seed
    # will be overridden
    sorted_reqs_params = zip(steps_add_reqs, seqs_max_tokens, prompts_lengths)
    requests: deque[tuple[int, EngineCoreRequest]] = deque()

    # seeds for random (repeated) prompts generation to test prefix caching
    if seeds:
        assert from_model_vocab, "when providing seeds we create random prompts"
        assert len(seeds) == len(steps_add_reqs), (
            "number of seeds must be equal to the number of prompts"
        )
    else:
        seeds = [None] * len(steps_add_reqs)

    for i, (add_step, max_tokens, prompt_length) in enumerate(sorted_reqs_params):
        # ignoring eos because we want to force the decoding to finish
        # after max_tokens exactly
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=0.0, logprobs=0, ignore_eos=True
        )
        request = create_random_request(
            request_id=i,
            num_tokens=prompt_length,
            sampling_params=sampling_params,
            model=model,
            from_model_vocab=from_model_vocab,
            seed=seeds[i],
        )
        requests.append((add_step, request))
        # NOTE: It is going to be decoded later
        generated_prompts.append(request.prompt_token_ids)

    return generated_prompts, requests


def dummy_assert_func(engine_core: EngineCore, step_ref: dict[str, Any], disable_asserts: bool):
    pass


@deprecated("This function is deprecated. Use validate_scheduler_steps instead.")
def check_scheduler_inference_steps(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
    max_batch_tkv_limit: int = -1,
    max_num_batched_tokens: int = None,
    random_prompts: bool = False,
    prefix_caching: bool = False,
    seeds: list[int] = None,
    extra_assert_funcs: Iterable[Callable[[EngineCore, dict[str, Any], bool], Any]] = [
        dummy_assert_func
    ],
):
    """
    Test the scheduler execution by comparing the scheduler attributes at each
    step with the provided reference values in 'checked_steps'.

    The missing steps from 'checked_steps' are automatically generated as decode
    steps, based on the existing elements in the list. For that to work, all the
    prefill steps and the first decode step after them needs be added to
    'checked_steps'
    """

    # Input parameters sanity check, not actual testing
    # ------
    if not (
        len(prompts_lengths) == len(seqs_max_tokens) and len(prompts_lengths) == len(steps_add_reqs)
    ):
        raise ValueError("Number of prompts should be consistent with number of max tokens.")

    if not (steps_add_reqs == sorted(steps_add_reqs) and steps_add_reqs[0] == 0):
        raise ValueError(
            "The list of steps where requests are added should be increasing start with 0"
        )

    if not (
        checked_steps == sorted(checked_steps, key=lambda x: x["step"])
        and len(checked_steps) == len(set(x["step"] for x in checked_steps))
    ):
        raise ValueError("List of checked steps needs to be of increasing order of step")
    # ------

    prompts, requests = generate_prompts(
        model,
        steps_add_reqs,
        seqs_max_tokens,
        prompts_lengths,
        from_model_vocab=random_prompts,
        seeds=seeds,
    )

    hf_results = generate_hf_output(
        model=model,
        prompts=prompts,
        max_new_tokens=seqs_max_tokens,
        ignore_eos=True,
    )

    abs_tol = ISCLOSE_ABS_TOL_QUANTIZATION if model.is_quantized else ISCLOSE_ABS_TOL
    # inject expectation.
    # json is fine to transfer between vllm subprocesses using pickle
    for idx, (req, hf) in enumerate(zip(requests, hf_results)):
        req[1].sampling_params.extra_args = {
            "golden_token_injector": {
                "expected_token_ids": hf["token_ids"],
                "expected_logprobs": hf["logprobs"],
                "error_threshold": abs_tol,
                "label": f"#{idx}",
            }
        }

    requests = [
        SchedulerTestRequest(add_step=r[0], request=r[1], hf_output=hf_results[i])
        for i, r in enumerate(requests)
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=requests,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_batch_tkv_limit=max_batch_tkv_limit,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=prefix_caching,
        extra_assert_funcs=extra_assert_funcs,
    )


def validate_scheduler_steps(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    requests: list[SchedulerTestRequest],
    checked_steps: list[dict[str, Any]],
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
    max_batch_tkv_limit: int = -1,
    max_num_batched_tokens: int = None,
    prefix_caching: bool = False,
    extra_assert_funcs: Iterable[Callable[[EngineCore, dict[str, Any], bool], Any]] = [
        dummy_assert_func
    ],
):
    """
    Creates a vllm.v1.engine and runs it step-by-step for the provided requests.
    Validates that the engine state matches the state given in `checked_steps`,
    and validates that the resulting output logprobs for each request is within
    an acceptable tolerance of hf output.
    """
    assert len(requests) == len(set(r.request.request_id for r in requests)), (
        "duplicate request IDs detected"
    )

    prompts = [r.request.prompt_token_ids for r in requests]
    hf_results = [r.hf_output for r in requests]

    # Setup the engine
    engine_core: EngineCore = get_cached_engine(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        use_pc=prefix_caching,
        available_blocks=available_blocks,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    scheduler: ChunkedPrefillSpyreScheduler = engine_core.scheduler

    tokenizer = get_tokenizer(model.name, revision=model.revision)

    # Override the TKV limit in the scheduler if needed
    if max_batch_tkv_limit >= 0:
        scheduler.max_batch_tkv_limit = max_batch_tkv_limit
    else:
        # This default value is set by platform.py
        scheduler.max_batch_tkv_limit = int(os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"))

    scheduler.do_interleaving = bool(int(os.getenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "1")))

    # In-between steps are added as normal decode steps
    checked_steps = augment_checked_steps(checked_steps)

    collected_outputs = defaultdict(
        lambda: {"token_ids": [], "logprobs": [], "text": "", "tokens": []}
    )

    # Run steps, until last step from 'checked_steps' is reached
    request_outputs = []
    requested_blocks, reserved_blocks = {}, {}
    for step in range(checked_steps[-1]["step"] + 1):
        # Add requests for this step
        while requests and requests[0].add_step == step:
            engine_core.add_request(requests.pop(0).request)

        # Check step if it is in the provided list of steps to check
        if checked_steps and step == checked_steps[0]["step"]:
            step_ref = checked_steps.popleft()

            waiting = [r.request_id for r in scheduler.waiting]
            running = [r.request_id for r in scheduler.running]
            out_reqs_ids = [r.request_id for r in request_outputs]
            out_reqs_finished = [r.request_id for r in request_outputs if r.finished]

            assert DISABLE_ASSERTS or (scheduler.tkv == step_ref["tkv"]), (
                f"Step {step}, tkv: {scheduler.tkv}"
            )
            assert DISABLE_ASSERTS or waiting == step_ref["waiting"], (
                f"Step {step}, waiting: {waiting}"
            )
            assert DISABLE_ASSERTS or running == step_ref["running"], (
                f"Step {step}, running: {running}"
            )
            assert DISABLE_ASSERTS or (out_reqs_ids == step_ref["request_outputs"]), (
                f"Step {step}, request outputs: {out_reqs_ids}"
            )

            ref_finished_reqs = step_ref.get("finished_requests", [])
            assert DISABLE_ASSERTS or (out_reqs_finished == ref_finished_reqs), (
                f"Step {step}, finished request output: {out_reqs_finished}"
            )

            # checking the scheduler handling of free and reserved blocks
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner

            n_blocks = scheduler.cache_config.num_gpu_blocks
            assert (
                scheduler.cache_config.num_gpu_blocks
                == scheduler.cache_config.num_gpu_blocks_override
            )
            block_size = model_runner.block_size
            n_reserved_blocks = (
                n_blocks - scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            )

            kv_cache_manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]

            req_ids2blocks = {
                req_id: [block.block_id for block in blocks]
                for req_id, blocks in kv_cache_manager.req_to_blocks.items()
                if blocks
            }
            # Account for blocks reused via prefix caching
            used_blocks = set()
            for blocks in req_ids2blocks.values():
                used_blocks = used_blocks.union(blocks)
            n_used_blocks = len(used_blocks)

            n_cached_blocks = n_prefix_hits = 0
            if prefix_caching:
                reqs = model_runner.requests
                prefix_hits = [
                    reqs[r_id].usable_blocks * block_size > reqs[r_id].num_computed_tokens
                    for r_id in req_ids2blocks
                ]
                cached_blocks = [reqs[r_id].usable_blocks for r_id in req_ids2blocks]
                n_cached_blocks = sum(cached_blocks)
                n_prefix_hits = sum(prefix_hits)

            if step > 0:
                if DISABLE_ASSERTS:
                    print(
                        f"{step=}, {n_reserved_blocks=}, {n_used_blocks=}, "
                        f"{scheduler.tkv=}, {waiting=}, {out_reqs_finished=}, "
                        f"{running=}, {out_reqs_ids=}, {n_prefix_hits=}, "
                        f"{n_cached_blocks=}"
                    )
                assert (
                    DISABLE_ASSERTS
                    or "n_reserved_blocks" not in step_ref
                    or (n_reserved_blocks == step_ref["n_reserved_blocks"])
                ), f"Step {step}, n_reserved_blocks: {n_reserved_blocks}"

                assert DISABLE_ASSERTS or (n_used_blocks == step_ref["n_used_blocks"]), (
                    f"Step {step}, n_used_blocks: {n_used_blocks}"
                )
                assert (
                    DISABLE_ASSERTS
                    or "n_prefix_hits" not in step_ref
                    or (n_prefix_hits == step_ref["n_prefix_hits"])
                ), f"Step {step}, n_prefix_hits: {n_prefix_hits}"
                assert (
                    DISABLE_ASSERTS
                    or "n_cached_blocks" not in step_ref
                    or (n_cached_blocks == step_ref["n_cached_blocks"])
                ), f"Step {step}, n_cached_blocks: {n_cached_blocks}"

            for extra_assert_func in extra_assert_funcs:
                extra_assert_func(engine_core, step_ref, DISABLE_ASSERTS)

        # last step: check that sequences used all their reserved blocks
        # Note: no early stopping, all sequences produce max_num_tokens
        if len(checked_steps) == 0:
            for req_id in requested_blocks:
                assert DISABLE_ASSERTS or requested_blocks[req_id] == reserved_blocks[req_id]

        # Perform next step
        step_output = engine_core.step()
        engine_core_output = step_output[0].get(0)
        request_outputs = engine_core_output.outputs if engine_core_output is not None else []

        for output in request_outputs:
            new_token_ids = output.new_token_ids
            new_logprobs = output.new_logprobs.logprobs
            assert DISABLE_ASSERTS or len(new_token_ids) == 1 and len(new_logprobs) == 1

            collected_outputs[output.request_id]["token_ids"].append(new_token_ids[0])
            collected_outputs[output.request_id]["logprobs"].append(new_logprobs[0][0])
            collected_outputs[output.request_id]["tokens"].append(
                tokenizer.decode(new_token_ids[0])
            )

    for k in collected_outputs:
        collected_outputs[k]["text"] = tokenizer.decode(collected_outputs[k]["token_ids"])
    output_keys = sorted(int(k) for k in collected_outputs)
    assert DISABLE_ASSERTS or output_keys[0] == 0 and output_keys[-1] == len(output_keys) - 1

    # convert dict of dicts to ordered list and make values immutable
    vllm_results = []
    for k in output_keys:
        output = collected_outputs[str(k)]
        for k, list_values in output.items():
            if isinstance(list_values, list):
                output[k] = tuple(list_values)
        vllm_results.append(output)

    compare_results(
        model=model,
        tensor_parallel_size=1,
        backend=backend,
        vllm_results=vllm_results,
        hf_results=hf_results,
        prompts=prompts,
    )
