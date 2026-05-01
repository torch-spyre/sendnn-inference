"""
This example shows how to run offline inference with a vision language model.

NOTE: At the moment, if you are checking parity, things may not line up
unless you compare eager against the FMS cpu model, i.e.,
    $ python vision_inference.py --backend eager --compare-target fms
"""

import argparse
import os
import platform
import time

import torch
from fms.models import get_model
from fms.utils import serialization
from fms.utils.generation import generate as fms_generate
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ibm-granite/granite-vision-3.3-2b")
parser.add_argument(
    "--max_model_len", "--max-model-len", type=int, default=8192
)  # one image has a max context of ~5k
parser.add_argument("--max_num_seqs", "--max-num-seqs", type=int, default=2)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--num-prompts", "-n", type=int, default=1)
parser.add_argument(
    "--max-tokens",
    type=str,
    default="8",
    help="Comma separated list of max tokens to use for each prompt. "
    "This list is repeated until prompts are exhausted.",
)
parser.add_argument("--backend", type=str, default="sendnn", choices=["eager", "sendnn"])

parser.add_argument(
    "--compare-target",
    type=str,
    default="fms",
    choices=["transformers", "fms"],
    help="Target to compare results against on CPU.",
)


def get_vllm_prompts(num_prompts, model_path):
    """Get the vLLM prompts to be processed."""
    # NOTE:
    # mistral-small-3.1 model has [IMG] as image token
    # llava-next has <image> as image token

    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    assert hasattr(processor, "image_token")

    image_token = processor.image_token

    template = f"<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n{image_token}\n{{}}\n<|assistant|>\n"  # noqa: E501

    images = [
        ImageAsset("cherry_blossom").pil_image,
        ImageAsset("stop_sign").pil_image,
    ]

    instructions = [
        "describe this image.",
        "what is shown in this image?",
        "what kind of flowers are these?",
    ]

    prompts = []
    for img in images:
        width, height = img.size
        for instr in instructions:
            # Make the images smol so that this example can run faster,
            # since we are not using a toy model here, and big images
            # can take up tons of tokens
            new_width = int(0.1 * width)
            new_height = int(0.1 * height)
            prompts.append(
                {
                    "prompt": template.format(instr),
                    "multi_modal_data": {
                        "image": img.resize((new_width, new_height)),
                    },
                }
            )

    prompts = prompts * (num_prompts // len(prompts) + 1)
    return prompts[:num_prompts], image_token


def compare_results(
    prompts: list[str],
    outputs_a: list[str],
    outputs_b: list[str],
    name_a: str,
    name_b: str,
    image_token: str,
):
    """Utils for comparing outputs from differing engines/implementations,
    e.g., transformers & vLLM.
    """

    print(f"Comparing {name_a} results with {name_b}")
    print("===============")
    any_differ = False
    for idx, (result_a, result_b) in enumerate(zip(outputs_a, outputs_b)):
        if result_a != result_b:
            img_tok_idx = prompts[idx].index(image_token)
            gen_prompt_idx = prompts[idx].index("<|assistant|>")
            raw_prompt = prompts[idx][img_tok_idx:gen_prompt_idx].strip()

            any_differ = True
            print(f"Results for prompt {idx} differ!")
            print(f"\nPrompt (no system/gen prompt):\n {repr(raw_prompt)}")
            print(f"\n{name_a} generated text:\n {result_a}\n")
            print(f"\n{name_b} generated text:\n {result_b}\n")
            print("-----------------------------------")

    if not any_differ:
        print("\nAll results match!\n")


### Alternate implementations to compare against
def get_transformers_results(model_path, vllm_prompts):
    """Process the results for HF Transformers running on CPU."""
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    return process_prompts(
        model_path,
        model,
        vllm_prompts,
        process_prompt_transformers,
    )


def process_prompt_transformers(model, max_tokens, inputs):
    """Process a single prompt using a transformers model."""
    return model.generate(**inputs, max_new_tokens=max_tokens)


def get_fms_results(model_path, vllm_prompts):
    """Process the results for FMS running on CPU."""

    model_config = AutoConfig.from_pretrained(model_path)

    kwargs = {}

    if model_config.model_type == "llava_next":
        # head_dim expansion required for granite vision
        serialization.extend_adapter(
            "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"]
        )

        kwargs = {
            "text_config": {"head_dim": 128},
            "override_hf_pretrained_config": True,
        }

    # Load, but don't compile (compare to CPU)
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.bfloat16,  # Matches default in vLLM for this model
        fused_weights=False,
        **kwargs,
    )

    return process_prompts(
        model_path,
        model,
        vllm_prompts,
        process_prompt_fms,
    )


def process_prompt_fms(model, max_tokens, inputs):
    """Process a single prompt using an FMS model."""
    input_ids = inputs.pop("input_ids")
    # May be better to use paged attn later on, but for now
    # we just use sdpa to avoid having to deal with padding
    # utils & position id management here
    inputs["attn_name"] = "sdpa_causal"

    return fms_generate(
        model,
        input_ids,
        max_new_tokens=max_tokens,
        use_cache=True,
        do_sample=False,  # Greedy decode
        extra_kwargs=inputs,
        prepare_model_inputs_hook=model.prepare_inputs_for_generation,
    )


def process_prompts(model_path, model, vllm_prompts, process_prompt):
    """Generic wrapper for running generate on either transformers or FMS."""
    processor = AutoProcessor.from_pretrained(model_path)
    num_prompts = len(vllm_prompts)
    generated_texts = []
    for i in range(num_prompts):
        # Prompts are preformatted, so don't worry about the chat template
        vllm_req = vllm_prompts[i]

        inputs = processor(
            text=vllm_req["prompt"],
            images=vllm_req["multi_modal_data"]["image"],
            return_tensors="pt",
        )
        # NOTE: Image tokens are expanded in the llava next preprocessor
        num_expanded_toks = inputs.input_ids.shape[1]

        target_output = process_prompt(
            model,
            max_tokens[i],
            inputs,
        )

        out_toks = target_output[0][num_expanded_toks:]
        # Make sure not to include EOS, since vLLM
        # doesn't return them, but FMS might.
        generated_text = processor.decode(
            out_toks,
            skip_special_tokens=True,
        )
        generated_texts.append(generated_text)

    return generated_texts


if __name__ == "__main__":
    args = parser.parse_args()

    max_num_seqs = args.max_num_seqs  # defines the max batch size

    if platform.machine() == "arm64":
        print(
            "Detected arm64 running environment. "
            "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
            "different version of the model using HF API which might not work "
            "locally on arm64."
        )
        os.environ["HF_HUB_OFFLINE"] = "1"

    if platform.system() == "Darwin":
        print("Setting VLLM_WORKER_MULTIPROC_METHOD=spawn to avoid forking problems on Mac OS")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    os.environ["SENDNN_INFERENCE_DYNAMO_BACKEND"] = args.backend

    prompts, image_token = get_vllm_prompts(args.num_prompts, args.model)

    # Set differing max_tokens so that the requests drop out of the batch at
    # different times
    max_tokens = [int(v) for v in args.max_tokens.split(",")]
    max_tokens = max_tokens * (args.num_prompts // len(max_tokens) + 1)
    max_tokens = max_tokens[: args.num_prompts]

    sampling_params = [
        SamplingParams(max_tokens=m, temperature=0.0, ignore_eos=True) for m in max_tokens
    ]

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=args.tp,
        limit_mm_per_prompt={"image": 1},  # Required for multimodal models
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    print("=============== GENERATE")
    t0 = time.time()
    vllm_outputs = llm.generate(prompts, sampling_params)
    vllm_results = [x.outputs[0].text for x in vllm_outputs]  # raw texts
    raw_prompts = [prompt["prompt"] for prompt in prompts]

    compare_target_map = {
        "transformers": get_transformers_results,
        "fms": get_fms_results,
    }

    # Since we always compare the results here, we don't bother
    # printing the raw results yet, since the head_dim patch
    # in FMS init tends to flood the logs anyway.
    cpu_results = compare_target_map[args.compare_target](
        model_path=args.model,
        vllm_prompts=prompts,
    )

    compare_results(
        prompts=raw_prompts,
        outputs_a=cpu_results,
        outputs_b=vllm_results,
        name_a=f"{args.compare_target} [cpu]",
        name_b="vllm [spyre]",
        image_token=image_token,
    )
