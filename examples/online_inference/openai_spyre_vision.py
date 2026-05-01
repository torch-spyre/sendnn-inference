""" 
This example shows how to use Spyre with vLLM for running online inference,
using granite vision. Note that currently, multimodal is *not* supported for
static baching.

First, start the server with the following command:

SENDNN_INFERENCE_DYNAMO_BACKEND=<your backend, e.g., sendnn/eager> \
vllm serve 'ibm-granite/granite-vision-3.3-2b' \
    --max-model-len=16384 \
    --max-num-seqs=2

NOTE: in the max feature case, a single image for granite vision can take
around 5k tokens, so keep this in mind when setting the max model length.
Also this script does *not* submit multiple requests as a batch.
This is because multimodal inputs are only supported for chat completions,
not completions, and the chat completions endpoint does not support batched
inputs.
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
    default=8,
    help="Maximum new tokens.",
)
parser.add_argument(
    "--num_prompts",
    type=int,
    default=4,
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


def get_vllm_prompts(num_prompts):
    """Get the vLLM prompts to be processed."""
    img_urls = [
        "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # noqa: E501
        "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",  # noqa: E501
    ]

    instructions = [
        "describe this image.",
        "what is shown in this image?",
        "are there any animals in this image?",
    ]

    prompts = []
    for img_url in img_urls:
        for instr in instructions:
            prompts.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instr},
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                    ],
                }
            )

    prompts = prompts * (num_prompts // len(prompts) + 1)
    return prompts[:num_prompts]


models = client.models.list()
model = models.data[0].id

prompts = get_vllm_prompts(args.num_prompts)

for prompt in prompts:
    stream = args.stream

    print(f"Prompt: {prompt}")
    start_t = time.time()

    chat_completion = client.chat.completions.create(
        messages=[prompt],
        model=model,
        max_completion_tokens=args.max_tokens,
        stream=stream,
    )

    end_t = time.time()
    print("Results:")
    if stream:
        for c in chat_completion:
            print(c.choices[0].delta.content, end="")
    else:
        print(chat_completion.choices[0].message.content)

    total_t = end_t - start_t
    print(f"\nDuration: {total_t}s")
