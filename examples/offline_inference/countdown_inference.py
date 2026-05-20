"""
This example shows how to run offline inference with countdown prompts.
All parameters are hardcoded for the granite-3.3-8b-instruct model.
"""

import argparse
import os
import platform
import time

import re


from vllm import LLM, SamplingParams


def validate_countdown(generated_text: str, prompt: str) -> dict:
    """
    Validate that the generated text contains properly decremented integers.
    
    Returns a dict with validation results:
    - is_valid: bool
    - expected_start: int (next number after prompt)
    - actual_numbers: list of extracted integers
    - issues: list of validation issues found
    """
    # Extract numbers from prompt to determine expected starting point
    prompt_numbers = [int(x) for x in re.findall(r'\d+', prompt)]
    if not prompt_numbers:
        return {
            "is_valid": False,
            "expected_start": None,
            "actual_numbers": [],
            "issues": ["No numbers found in prompt"]
        }
    
    expected_start = prompt_numbers[-1] - 1
    
    # Extract numbers from generated text
    generated_numbers = [int(x) for x in re.findall(r'\d+', generated_text)]
    
    issues = []
    is_valid = True
    
    if not generated_numbers:
        issues.append("No numbers found in generated text")
        is_valid = False
    else:
        # Check if first number matches expected
        if generated_numbers[0] != expected_start:
            issues.append(f"Expected first number to be {expected_start}, got {generated_numbers[0]}")
            is_valid = False
        
        # Check if sequence is properly decremented
        for i in range(len(generated_numbers) - 1):
            if generated_numbers[i] - generated_numbers[i + 1] != 1:
                issues.append(
                    f"Non-consecutive numbers at position {i}: "
                    f"{generated_numbers[i]} -> {generated_numbers[i + 1]}"
                )
                is_valid = False
    
    return {
        "is_valid": is_valid,
        "expected_start": expected_start,
        "actual_numbers": generated_numbers,
        "issues": issues
    }

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run offline inference with countdown prompts"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["eager", "sendnn"],
        default="sendnn",
        help="Backend to use for inference (default: sendnn)"
    )
    args = parser.parse_args()
    
    # Hardcoded configuration
    MODEL = "ibm-granite/granite-3.3-8b-instruct"
    MAX_MODEL_LEN = 4096
    MAX_NUM_SEQS = 4
    TENSOR_PARALLEL_SIZE = 4
    MAX_TOKENS = 300
    ENABLE_PREFIX_CACHING = True
    MAX_NUM_BATCHED_TOKENS = 512
    BACKEND = args.backend

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

    os.environ["SENDNN_INFERENCE_DYNAMO_BACKEND"] = BACKEND
    os.environ["SENDNN_INFERENCE_CP_INTERLEAVE_STEPS"] = "0"

    print(f"Backend: {BACKEND}")
    
    # Create countdown prompts with actual countdown values
    prompts = [
        " ".join(str(i) for i in range(200, 180, -1)) + " ",
        " ".join(str(i) for i in range(150, 130, -1)) + " ",
        " ".join(str(i) for i in range(1000, 980, -1)) + " ",
        " ".join(str(i) for i in range(500, 480, -1)) + " ",
    ]

    print(f"Model: {MODEL}")
    print(f"Max Model Length: {MAX_MODEL_LEN}")
    print(f"Max Num Seqs: {MAX_NUM_SEQS}")
    print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    print(f"Number of prompts: {len(prompts)}")
    print("=" * 80)
    
    # Print the prompts
    print("\nPROMPTS:")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}:")
        print(prompt)
    print("\n" + "=" * 80)

    # Create sampling parameters
    sampling_params = [
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True)
        for _ in prompts
    ]

    # Create an LLM
    print("Initializing LLM...")
    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )

    # Generate texts from the prompts
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.time() - t0
    
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(f"\nTime elapsed for {total_tokens} tokens is {elapsed_time:.2f} sec")
    print(f"Tokens per second: {total_tokens / elapsed_time:.2f}")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    all_valid = True
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        
        # Validate countdown
        validation = validate_countdown(generated_text, prompt)
        
        print(f"\n[Prompt {i}]")
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Tokens generated: {num_tokens}")
        print(f"\nValidation:")
        print(f"  Status: {'✓ PASS' if validation['is_valid'] else '✗ FAIL'}")
        print(f"  Expected start: {validation['expected_start']}")
        print(f"  Numbers found: {len(validation['actual_numbers'])}")
        if validation['actual_numbers']:
            print(f"  First few numbers: {validation['actual_numbers'][:10]}")
            print(f"  Last few numbers: {validation['actual_numbers'][-10:]}")
        if validation['issues']:
            print(f"  Issues:")
            for issue in validation['issues']:
                print(f"    - {issue}")
        
        if not validation['is_valid']:
            all_valid = False
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print(f"OVERALL VALIDATION: {'✓ ALL PASSED' if all_valid else '✗ SOME FAILED'}")
    print("=" * 80)