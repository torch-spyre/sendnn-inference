"""
This example exercise long context lengths

Let's say you want to test the following configuration

Prefill: Max_prompt = 4K, prefill batch-size = 1.
Generation: Max_context = 8K, Max_batch = 4.

Then the command line will be

```
python long_context.py --max-num-seqs 4 --max-prompt-len 4096 \
        --max-model-len 8192 
```

To compare with cpu, add `--compare-with-cpu`.

All sequences will run up to the max context length.

"""

import argparse
import os
import platform
import sys
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ibm-ai-platform/micro-g3.3-8b-instruct-1b")
    parser.add_argument("--max_model_len", "--max-model-len", type=int, default=2048)
    parser.add_argument("--max_prompt_len", "--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max_num_seqs", "--max-num-seqs", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-prompts", "-n", type=int, default=8)
    parser.add_argument("--compare-with-cpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--trunc_print_len", "--trunc-print-len", type=int, required=False)
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--backend", type=str, default="sendnn", choices=["eager", "sendnn"])

    args = parser.parse_args()

    trunc = args.trunc_print_len

    assert args.max_prompt_len <= args.max_model_len

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

    template = "Summarize the following code: \n\n{}"

    def get_python_file(source_file):
        for path in sys.path:
            file_path = os.path.join(path, source_file)
            if os.path.isfile(file_path):
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
        raise Exception(f"File {source_file} not found")

    example_files = [
        "os.py",
        "gzip.py",
        "inspect.py",
        "abc.py",
        "dataclasses.py",
        "enum.py",
        "functools.py",
        "io.py",
    ]

    file_contents = [get_python_file(e) for e in example_files]

    prompts = [template.format(c) for c in file_contents]

    prompts = prompts * (args.num_prompts // len(prompts) + 1)
    prompts = prompts[0 : args.num_prompts]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenized_prompts = tokenizer(prompts)["input_ids"]
    tokenized_prompts = [p[: args.max_prompt_len] for p in tokenized_prompts]

    prompt_lens = [len(p) for p in tokenized_prompts]

    max_prompt = max(prompt_lens)
    min_prompt = min(prompt_lens)

    if max_prompt < args.max_prompt_len:
        print(f"Warning, none of the prompts reach the maximum length({args.max_prompt_len})")

    print(f"All prompts have lengths between {min_prompt} and {max_prompt}")

    def round_up(t):
        return ((t + 63) // 64) * 64

    tokens_to_generate = [args.max_model_len - round_up(prompt_len) for prompt_len in prompt_lens]

    sampling_params = [
        SamplingParams(max_tokens=t, temperature=0.0, ignore_eos=True) for t in tokens_to_generate
    ]

    vllm_token_prompts = [TokensPrompt(prompt_token_ids=p) for p in tokenized_prompts]

    # Create an LLM.
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    print("=============== GENERATE")
    t0 = time.time()
    outputs = llm.generate(vllm_token_prompts, sampling_params)
    print("Time elapsed for all prompts is %.2f sec" % (time.time() - t0))
    print("===============")
    for output, prompt in zip(outputs, prompts):
        generated_text = output.outputs[0].text[:trunc]
        prompt = prompt[:trunc]
        print(f"\nPrompt:\n {prompt!r}")
        print(f"\nGenerated text (truncated):\n {generated_text!r}\n")
        print("-----------------------------------")

    if args.compare_with_cpu:
        print("Comparing results with HF on cpu")
        print("===============")
        any_differ = False

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model)

        for i in range(args.num_prompts):
            prompt = prompts[i]

            hf_input_tokens = torch.tensor(tokenized_prompts[i]).unsqueeze(0)
            hf_output = model.generate(
                hf_input_tokens,
                do_sample=False,
                min_new_tokens=tokens_to_generate[i],
                max_new_tokens=tokens_to_generate[i],
                return_dict_in_generate=True,
                output_scores=True,
            )

            # decode output tokens after first removing input tokens (prompt)
            hf_generated_text = tokenizer.batch_decode(
                hf_output.sequences[:, len(hf_input_tokens[0]) :]
            )[0]

            if hf_generated_text != outputs[i].outputs[0].text:
                any_differ = True
                spyre_output = outputs[i].outputs[0].text
                print(f"Results for prompt {i} differ on cpu")
                print(f"\nPrompt:\n {prompt[:trunc]!r}")
                print(f"\nSpyre generated text:\n {spyre_output[:trunc]!r}\n")
                print(f"\nCPU generated text:\n {hf_generated_text[:trunc]!r}\n")
                print("-----------------------------------")

        if not any_differ:
            print("\nAll results match!\n")
