"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

import pytest
from output_util import validate_vllm_vs_hf_output, kwargs_for_mode, generate_spyre_vllm_output
from spyre_util import (
    ModelInfo,
    get_chicken_soup_prompts,
    skip_unsupported_tp_size,
)
from vllm import SamplingParams, LLM


@pytest.mark.full_model
@pytest.mark.basic
# `mode` here is parametrized directly so that we can use the `cp` mode that disables prefix caching
# This mode is turned off by default in conftest.py, this is the one test that will ensure vllm
# boots with prefix caching disabled.
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("pc", marks=pytest.mark.prefix_caching, id="pc"),
        pytest.param("cp", marks=pytest.mark.chunked_prefill, id="cp"),
    ],
)
def test_output(
    model: ModelInfo,
    tp_size: int,
    backend: str,
    mode: str,
    max_num_seqs: int,
    max_model_len: int,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
    runtime_xfail,
) -> None:
    """
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.

    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 4
        * tensor parallelism: 1, 2, 4, 8
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: 20 (same for all the prompts)
    """

    skip_unsupported_tp_size(tp_size, backend)

    prompts = get_chicken_soup_prompts(4)

    max_new_tokens = 4

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        max_num_seqs=max_num_seqs,
        **kwargs_for_mode(mode),
    )


def test_batch_handling(
    model: ModelInfo,
    backend: str,
    mode: str,
    max_num_seqs: int,
    max_model_len: int,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
):
    """Test that the spyre worker correctly handles
    continuous batches of requests that
    finish after different numbers of forward passes

    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 2
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: [5, 20, 10, 5]
    """

    prompts = get_chicken_soup_prompts(4)
    max_new_tokens = [5, 20, 10, 5]
    vllm_sampling_params = [
        SamplingParams(
            max_tokens=max_new_tokens[i],
            min_tokens=max_new_tokens[i],
            temperature=0,
            ignore_eos=True,
            logprobs=0,
        )
        for i in range(len(max_new_tokens))
    ]

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        max_model_len=max_model_len,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_new_tokens=max_new_tokens,
        max_num_seqs=max_num_seqs,
        **kwargs_for_mode(mode),
    )


@pytest.mark.parametrize("backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_max_tokens(
    model: ModelInfo,
    backend: str,
    max_model_len: int,
    max_num_seqs: int,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
    mode: str,
):
    """Test that batches of requests that are longer than the `max_model_len` are correctly
    rejected"""
    max_tokens = 20

    overflow_prompt = " ".join(["a"] * max_model_len)

    vllm_sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=0, ignore_eos=True, logprobs=0
    )

    # The text of the error raised by vllm changed from 0.11.0 to 0.11.1
    with pytest.raises(ValueError, match="(max model context length|maximum model length)"):
        generate_spyre_vllm_output(
            model=model,
            prompts=[overflow_prompt],
            max_model_len=max_model_len,
            sampling_params=vllm_sampling_params,
            tensor_parallel_size=1,
            backend=backend,
            max_num_seqs=max_num_seqs,
            monkeypatch=monkeypatch,
            **kwargs_for_mode(mode),
        )


@pytest.mark.prefix_caching
@pytest.mark.parametrize("backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_tkv_limits_checked_correctly_on_prefix_hits(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that we don't overflow tkv limits when we have a prefix hit"""
    monkeypatch.setenv("VLLM_DT_MAX_BATCH_TKV_LIMIT", "2048")
    monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", backend)

    llm = LLM(
        model=model.name,
        max_model_len=1024,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        revision=model.revision,
    )

    base_prompt = "0 1 2 3 4 5 6 7 8 9 " * 24
    prompts = [base_prompt] * 5
    prompts.append(base_prompt * 2)

    llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(max_tokens=32),
    )
