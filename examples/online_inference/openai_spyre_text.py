""" 
This example shows how to use Spyre with vLLM for running online inference.

Continuous Batching:

First, start the server with the following command:
    vllm serve 'ibm-granite/granite-3.3-8b-instruct' \
        --max-model-len=2048 \
        --tensor-parallel-size=4 \
        --max-num-seqs=4

This sets up a server with max batch size 4. This allows vllm to process up to four prompts at once,
which you can do by running this script with `--batch_size` > 1.
"""

import argparse
import time

from openai import OpenAI

parser = argparse.ArgumentParser(
    description="Script to submit an inference request to vllm server."
)

parser.add_argument(
    "--max_tokens",
    type=int,
    default=20,
    help="Maximum new tokens.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
)
parser.add_argument(
    "--num_prompts",
    type=int,
    default=3,
)
parser.add_argument(
    "--stream",
    action=argparse.BooleanOptionalAction,
    help="Whether to stream the response.",
)

args = parser.parse_args()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:"
)

instructions = [
    "Provide a list of instructions for preparing chicken soup for a family" + " of four.",
    "Please compare New York City and Zurich and provide a list of" + " attractions for each city.",
    "Provide detailed instructions for preparing asparagus soup for a" + " family of four.",
]

prompts = [template.format(instr) for instr in instructions]
prompts = prompts * (args.num_prompts // len(prompts) + 1)
prompts = prompts[0 : args.num_prompts]

# This batch size must match SENDNN_INFERENCE_WARMUP_BATCH_SIZES
batch_size = args.batch_size
print("submitting prompts of batch size", batch_size)

# making sure not to submit more prompts than the batch size
for i in range(0, len(prompts), batch_size):
    prompt = prompts[i : i + batch_size]

    stream = args.stream

    print(f"Prompt: {prompt}")
    start_t = time.time()

    completion = client.completions.create(
        model=model,
        prompt=prompt,
        echo=False,
        n=1,
        stream=stream,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    end_t = time.time()
    print("Results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)

    total_t = end_t - start_t
    print(f"Duration: {total_t}s")
