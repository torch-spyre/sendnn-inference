"""Smoke test for `--load-format dummy` (random init, no checkpoint load)."""

import pytest
from spyre_util import ModelInfo, get_chicken_soup_prompts
from vllm import LLM, SamplingParams

from sendnn_inference import envs as envs_spyre


@pytest.mark.cpu
def test_load_format_dummy_generates_tokens(
    model: ModelInfo,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    """The dummy load path random-inits weights and still produces tokens
    (output content is meaningless and not checked)."""
    envs_spyre.override("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager")

    prompts = get_chicken_soup_prompts(1)
    max_new_tokens = 4

    llm = LLM(
        model=model.name,
        revision=model.revision,
        load_format="dummy",
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == max_new_tokens
