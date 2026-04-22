"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_mm.py`.
"""

from fms.utils.generation import generate as fms_generate
from fms.models import get_model

from transformers import AutoProcessor, AutoConfig

import pytest
from output_util import generate_spyre_vllm_output
from spyre_util import get_single_image_prompts, get_spyre_model_list
import torch
from vllm import SamplingParams

# Ensure the llava next mm mapping is imported, since
# the FMS serialization utilities are patched at import time,
# and the patching is currently NOT idempotent.
import sendnn_inference.multimodal.mm_mappings.llava_next  # noqa: F401

# We should not use a very large value here, because
# we do not have tiny multimodal models at the moment.
MAX_TOKENS = 8


def generate_fms_results(processor, model_path, prompts):
    # Ensure the llava next util has been imported, as we fix
    # the FMS serialization as a side effect at import time
    config_dict = {"head_dim": 128}

    # Load, but don't compile (compare to CPU)
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.bfloat16,  # Matches default in vLLM for this model
        fused_weights=False,
        override_hf_pretrained_config=True,
        text_config=config_dict,
    )

    generated_texts = []
    for mm_prompt in prompts:
        inputs = processor(
            text=mm_prompt["prompt"],
            images=mm_prompt["multi_modal_data"]["image"],
            return_tensors="pt",
        )

        num_expanded_toks = inputs.input_ids.shape[1]
        input_ids = inputs.pop("input_ids")
        # May be better to use paged attn later on, but for now
        # we just use sdpa to avoid having to deal with padding
        # utils & position id management here
        inputs["attn_name"] = "sdpa_causal"

        res = fms_generate(
            model,
            input_ids,
            max_new_tokens=MAX_TOKENS,
            use_cache=True,
            do_sample=False,  # Greedy decode
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
        )
        out_toks = res.squeeze()
        fms_texts = processor.decode(
            out_toks[num_expanded_toks:],
            skip_special_tokens=True,
        )
        generated_texts.append(fms_texts)
    return generated_texts


@pytest.mark.skip("Multimodal E2E tests are currently disabled; no tiny model")
@pytest.mark.cpu
@pytest.mark.parametrize("model", get_spyre_model_list(isMultimodal=True))
def test_alignment_with_fms(model, mode, monkeypatch):
    # Only run continuous batching with chunked prefill for now (slow tests)
    if mode != "cp":
        pytest.skip("Only running multimodal tests for chunked prefill")

    processor = AutoProcessor.from_pretrained(model.name)
    hf_config = AutoConfig.from_pretrained(model.name)
    image_token = processor.decode(hf_config.image_token_index)

    prompts = get_single_image_prompts(
        1,
        image_token,
        tile_size=hf_config.vision_config.image_size,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        ignore_eos=True,
        logprobs=0,
    )

    # NOTE: It would be more ideal to directly validate the outputs of vLLM and HF,
    # but there seem to be parity problems and we see very seemingly very different
    # values for logprobs :( for now we compare against FMS instead.
    # TODO (Alex): Investigate this and align with HF so that we
    # can leverage expected response caching in the tests!
    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        backend="eager",
        max_num_seqs=1,
        monkeypatch=monkeypatch,
        max_model_len=2048,
        max_num_batched_tokens=1024 if mode == "cp" else None,
    )

    fms_texts = generate_fms_results(processor, model.name, prompts)
    # Compare the newly decoded texts with FMS
    # and sendnn_inference running with the eager backend.
    for fms_text, vllm_result in zip(fms_texts, vllm_results):
        assert vllm_result["text"] == fms_text
