"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

import asyncio
from typing import cast
from unittest.mock import patch
import time

import pytest
import openai
import httpx
from llm_cache import get_llm
from llm_cache_util import force_engine_shutdown
from output_util import compare_results, extract_output, generate_hf_output, setup_golden_token
from pytest_mock.plugin import MockerFixture
from spyre_util import (
    ModelInfo,
    RemoteOpenAIServer,
    get_chicken_soup_prompts,
    get_longer_chicken_soup_prompts,
)
from vllm import LLM, SamplingParams

from scheduling_utils import random_prompt
from sendnn_inference.v1.worker.spyre_model_runner import SamplingForwardInputs


def get_model_runner(cp_model: LLM):
    return (
        cp_model.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    )


chicken_soup_prompts = get_longer_chicken_soup_prompts(4)

# NOTE: considering granite 3.3 tokenizer
# Should have 51 tokens
prompt_51 = get_chicken_soup_prompts(1)[0]
# Should have 95 tokens
prompt_95 = chicken_soup_prompts[0]
# Should have 251
prompt_251 = chicken_soup_prompts[0] + chicken_soup_prompts[1] + chicken_soup_prompts[2]
# Should have 260 tokens
prompt_260 = chicken_soup_prompts[0] + chicken_soup_prompts[2] + chicken_soup_prompts[3]

USE_CASES = {
    # Case I - Prompt fits in a single chunk
    "case_Ia": (prompt_95, 128, 1, 0),
    # Case I - Prompt fits in a single chunk with left padding
    "case_Ib": (prompt_51, 128, 1, 64),
    # Case II - Has left padding
    "case_II": (prompt_260, 128, 3, 64),
    # Case III again - no padding
    "case_III": (prompt_251, 128, 2, 0),
}


# The changes here to not cache this LLM do make it possible to run this test without --forked
# in some cases, but we still see crashes in CI. Marking it fork_required for our CI to fork it
# anyway.
@pytest.mark.fork_required
@pytest.mark.chunked_prefill
@pytest.mark.parametrize("use_case", list(USE_CASES.keys()))
def test_chunked_prefill_correctness(
    model: ModelInfo,
    backend: str,
    max_num_seqs: int,
    max_model_len: int,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    use_case: str,
    use_llm_cache,
) -> None:
    """
    Minimal test to check if sendnn-inference activate code for chunked prefill for
    a prompt greater than the chunk size
    """

    (prompt, chunk_size, expected_chunk_count, expected_left_padding) = USE_CASES[use_case]
    max_new_tokens = 8

    hf_outputs = generate_hf_output(
        model=model,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        ignore_eos=True,
    )
    ### NB: May not be guaranteed to be set
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Spyre chunked-prefill correctness cases cannot safely reuse the same
    # cached LLM across different use_case prompt shapes in one pytest process.
    use_cached_llm = backend != "sendnn"
    cp_model = get_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=chunk_size,
        cached=use_cached_llm,
    )
    try:
        model_runner = get_model_runner(cp_model)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, temperature=0, logprobs=0, ignore_eos=True
        )
        git_sampling_params = setup_golden_token(model, sampling_params, hf_outputs)

        _prepare_chunked_prefill = model_runner._prepare_chunked_prefill
        records = []

        def wrapper(self, *args, **kwargs):
            model_input = _prepare_chunked_prefill(self, *args, **kwargs)
            records.append(model_input)
            return model_input

        with patch.object(model_runner, "_prepare_chunked_prefill", wraps=wrapper) as spy:
            results = cp_model.generate(prompt, git_sampling_params)
            vllm_results = [extract_output(results[0])]

            for r in records:
                model_input = cast(SamplingForwardInputs, r)
                # Must be a single value
                left_padding = model_input.left_padded_prompt_mask[0].item()
                assert left_padding == expected_left_padding

        # Validate if the prefill was chunked
        spy.assert_called()
        assert spy.call_count == expected_chunk_count

        # Validate output
        compare_results(
            model=model,
            tensor_parallel_size=1,
            backend=backend,
            vllm_results=vllm_results,
            hf_results=hf_outputs,
            prompts=[prompt],
        )
    finally:
        if not use_cached_llm:
            force_engine_shutdown(cp_model)


@pytest.mark.parametrize("mode", [pytest.param("pc", marks=pytest.mark.prefix_caching, id="pc")])
@pytest.mark.asyncio
async def test_chunked_prefill_kv_cache_stats(
    remote_openai_server: RemoteOpenAIServer,
    model,
    backend,
    tp_size,
    mode,
    max_num_seqs,
    max_model_len,
    max_num_batched_tokens,
):
    assert max_num_batched_tokens == 128
    # Test that vllm metrics include prefix caching data
    client = remote_openai_server.get_async_client()

    prompt = random_prompt(
        model=model,
        seed=0,
        length=max_model_len // 2,  # try to span multiple chunks
    )

    # Start a bunch of duplicate requests so that we hit prefix cache:
    request_tasks: list[asyncio.Task] = []
    # 2x max batch size so that there's some queueing
    for _ in range(2 * max_num_seqs):
        request_tasks.append(
            asyncio.create_task(
                client.completions.create(model=model.name, prompt=prompt, max_tokens=5)
            )
        )

    # Wait for the first request to finish to ensure that requests are
    # processing
    requests_finished = 0
    while requests_finished < 1:
        done, pending = await asyncio.wait(request_tasks, return_when=asyncio.FIRST_COMPLETED)
        requests_finished += len(done)
        request_tasks = pending

    # Now that requests are processing, check metrics for KV cache usage.
    # This must be done while requests are in-flight, once they finish this
    # gauge will be 0.
    metrics = await get_metrics(client)
    kv_cache_usage = get_metric_value(metrics, "vllm:kv_cache_usage_perc")
    assert kv_cache_usage > 0

    # Wait for at least 2 requests to finish to ensure the prefix cache was used
    while requests_finished < 2:
        done, pending = await asyncio.wait(request_tasks, return_when=asyncio.FIRST_COMPLETED)
        requests_finished += len(done)
        request_tasks = pending

    # Cancel the rest
    for task in request_tasks:
        task.cancel()

    # Check the prefix cache counters
    # vLLM should be reporting these counters based on the last 1000 requests
    metrics = await get_metrics(client)
    total_tokens = get_metric_value(metrics, "vllm:prefix_cache_queries_total")
    hit_tokens = get_metric_value(metrics, "vllm:prefix_cache_hits_total")

    # Prefix cache rate won't be 100% because of chunking, but we should still
    # hit _some_ cache
    assert hit_tokens / total_tokens > 0.1


async def get_metrics(client: openai.AsyncOpenAI) -> list[str]:
    response = await client.get("../metrics", cast_to=httpx.Response)
    assert response.status_code == 200
    metrics = response.text
    return metrics.splitlines()


def get_metric_value(metrics: list[str], metric_name: str) -> float:
    metric_line = [line for line in metrics if line.startswith(metric_name)][0]
    return float(metric_line.split(" ")[-1])


async def reset_kv_cache(client: openai.AsyncOpenAI) -> list[str]:
    response = await client.post("../reset_prefix_cache", cast_to=httpx.Response)
    assert response.status_code == 200


@pytest.mark.parametrize("mode", [pytest.param("pc", marks=pytest.mark.prefix_caching, id="pc")])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize(
    "prompt_len, hit_len",
    [
        (53, 0),  # Less than one block, no cache hit
        (127, 0),  # Less than one chunk, no cache hit
        (144, 64),  # Two full blocks plus one partial. With one padding block, only one block can
        # be loaded from cache since the second chunk is already the last one and must
        # be recomputed.
        (
            250,
            128,
        ),  # 3 full blocks plus one partial. Since no padding is required, the entire first
        # chunk can be loaded from cache. The last chunk must be recomputed.
        (
            299,
            192,
        ),  # 4 full blocks plus one partial. This means that there will be one cached block
        # in the first chunk due to padding. The second chunk can be loaded 100%
        (350, 256),  # 5 full blocks plus 1 partial. Since no padding is required, the first to
        # chunks are loaded for cache.
        (420, 320),  # 6 full chunks plus 1 partial. Since there is padding, we can only load one
        # block from the first chunk. The second and third chunks are loaded completely
    ],
)
@pytest.mark.asyncio
async def test_max_prefix_hits(
    remote_openai_server: RemoteOpenAIServer,
    model,
    backend,
    tp_size,
    mode,
    prompt_len,
    hit_len,
    max_num_seqs,
    max_model_len,
    max_num_batched_tokens,
):
    assert max_num_batched_tokens == 128
    # Test that vllm metrics include prefix caching data
    client = remote_openai_server.get_async_client()

    await reset_kv_cache(client)

    metrics = await get_metrics(client)
    initial_hits = get_metric_value(metrics, "vllm:prefix_cache_hits_total")

    prompt = random_prompt(model=model, seed=time.time_ns(), length=prompt_len)

    await client.completions.create(model=model.name, prompt=prompt, max_tokens=1)
    await client.completions.create(model=model.name, prompt=prompt, max_tokens=1)

    metrics = await get_metrics(client)
    final_hits = get_metric_value(metrics, "vllm:prefix_cache_hits_total")
    hit_tokens = final_hits - initial_hits
    assert hit_tokens == hit_len


@pytest.mark.parametrize("mode", [pytest.param("pc", marks=pytest.mark.prefix_caching, id="pc")])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize(
    "prompt_len, hit_len",
    [
        # In this test, the first request is a prefix of the second, with 128 less tokens.
        # So for all prompt lengths below, subtract 128 tokens.
        (144, 0),  # Less than one block, no cache hit
        (250, 0),  # Less than one chunk, no cache hit
        (299, 64),  # Two full blocks plus one partial. With one padding block, only one block can
        # be loaded from cache since the second chunk is already the last one and must
        # be recomputed.
        (
            350,
            128,
        ),  # 3 full blocks plus one partial. Since no padding is required, the entire first
        # chunk can be loaded from cache. The last chunk must be recomputed.
        (
            420,
            192,
        ),  # 4 full blocks plus one partial. This means that there will be one cached block
        # in the first chunk due to padding. The second chunk can be loaded 100%
    ],
)
@pytest.mark.asyncio
async def test_partial_prefix_hits(
    remote_openai_server: RemoteOpenAIServer,
    model,
    backend,
    tp_size,
    mode,
    prompt_len,
    hit_len,
    max_num_seqs,
    max_model_len,
    max_num_batched_tokens,
):
    assert max_num_batched_tokens == 128
    # Test that vllm metrics include prefix caching data
    client = remote_openai_server.get_async_client()

    await reset_kv_cache(client)

    metrics = await get_metrics(client)
    initial_hits = get_metric_value(metrics, "vllm:prefix_cache_hits_total")

    prompt = random_prompt(model=model, seed=time.time_ns(), length=prompt_len)

    await client.completions.create(model=model.name, prompt=prompt, max_tokens=1)
    await client.completions.create(
        model=model.name, prompt=prompt[:-max_num_batched_tokens], max_tokens=1
    )

    metrics = await get_metrics(client)
    final_hits = get_metric_value(metrics, "vllm:prefix_cache_hits_total")
    hit_tokens = final_hits - initial_hits
    assert hit_tokens == hit_len
