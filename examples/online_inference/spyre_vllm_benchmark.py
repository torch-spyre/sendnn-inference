#!/usr/bin/env python3
"""
Example usage:
python3 container-scripts/spyre_vllm_benchmark.py
--prompt-dir $HOME/prompts/
--tokenizer-dir $HOME/models/granite-3.3-8b-instruct
--output-dir $HOME/output/
--port 8000
--max-tokens 64
--min-tokens 64
"""

# Imports
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import requests
from openai import APIConnectionError, OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizer


# Classes
class InferenceResults(NamedTuple):
    outputs: list[str]
    inference_time: float
    output_token_count: int
    ttft: float
    token_latencies: list[list[float]]


# Functions
def parse_args():
    parser = argparse.ArgumentParser(description="SenDNN Inference inference benchmarking script.")
    parser.add_argument(
        "--prompt-dir", required=True, type=str, help="Path to directory containing .txt files"
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        type=str,
        help="Path to a directory containing a tokenizer",
    )
    parser.add_argument(
        "--port", required=False, help="Port of running container to connect to.", default=8000
    )
    parser.add_argument(
        "--max-tokens",
        required=False,
        type=int,
        help="Maximum number of tokens to generate",
        default=64,
    )
    parser.add_argument(
        "--min-tokens",
        required=False,
        type=int,
        help="Minimum number of tokens to generate",
        default=0,
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        help="Output directory to dump results and performance metrics",
        default=None,
    )
    return parser.parse_args()


def create_client(api_key: str, base_url: str) -> OpenAI:
    """
    Creates and returns an OpenAI client.

    Args:
        api_key (str): The OpenAI API key.
                       Often set to "EMPTY" for local inference setups.
        base_url (str): The base URL of the OpenAI-compatible API,
                        e.g., "http://localhost:8000/v1".

    Returns:
        OpenAI: An instance of the OpenAI client initialized with the provided
                API key and base URL.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client


def test_server_connection(client: OpenAI, endpoint: str) -> bool:
    """
    Tests the connection to a specified endpoint of the OpenAI server.

    Args:
        client (OpenAI): The OpenAI client instance.
        endpoint (str): The relative endpoint to test (e.g., "/models/").

    Returns:
        bool: True if the server responds with a 200 status code;
              False otherwise.
    """
    try:
        base_url = str(client.base_url).rstrip("/")
        response = requests.get(base_url + endpoint)
        return response.status_code == 200
    except requests.RequestException as e:
        print(e)
        return False


def connect(client: OpenAI, endpoint: str, max_tries: int = 5) -> None:
    """
    Attempts to connect to the specified server endpoint max_tries times.

    Args:
        client (OpenAI): The OpenAI client instance.
        endpoint (str): The relative endpoint to connect to (e.g., "/models").
        max_tries (int): Maximum number of connection attempts. Default is 5.

    Returns:
        None

    Raises:
        RuntimeError: If connection fails after max_tries attempts.
    """
    tries = 0
    while tries < max_tries:
        try:
            base_url = str(client.base_url).rstrip("/")
            address = base_url + endpoint
            response = requests.get(address)
            if response.status_code == 200:
                return
        except requests.RequestException as e:
            print(f"Connection attempt {tries + 1} failed: {e}")
        time.sleep(1)
        tries += 1
    raise RuntimeError(f"Failed to connect to {endpoint} after {max_tries} attempts.")


def get_tokenizer(model_path: str):
    """
    Loads and returns a tokenizer from the specified model path.

    Args:
        model_path (str): Path to the pretrained model directory
                          or identifier from Hugging Face Hub.

    Returns:
        PreTrainedTokenizer: A tokenizer instance loaded from model_path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_model_from_server(client: OpenAI) -> str:
    """
    Retrieves the first available model ID from the OpenAI-compatible server.

    Args:
        client (OpenAI): An instance of the OpenAI client.

    Returns:
        str: The ID of the first model returned by the server.

    Raises:
        SystemExit: If there is a connection error while fetching models.
    """
    model = None
    try:
        models = client.models.list()
        model = models.data[0].id
        print(f"Found Model: {model}")
    except APIConnectionError as e:
        print(f"Connection Error: {e}")
        exit(1)
    return model


def process_input_prompts(prompt_dir: str) -> list[Path]:
    """
    Collects all `.txt` prompt files from the specified directory.

    Args:
        prompt_dir (str): Path to the directory containing prompt files.

    Returns:
        list[Path]: List of Paths to the `.txt` prompt files found in
                    the directory.
    """
    prompt_list = list(Path(prompt_dir).glob("*.txt"))
    if not prompt_list:
        print(f"No .txt files found in {prompt_dir}.")
        exit(1)
    print(f"Found {len(prompt_list)} prompt files at {prompt_dir}")
    return prompt_list


def save_results(output_dir: Path, prompt_files: list[Path], model: str, results: InferenceResults):
    """
    Saves model inference outputs and performance metrics to the specified
    output directory.

    Each prompt's generated output is written to a separate text file named
    after the prompt, and performance metrics are written to a single
    `performance_metrics.txt` file.

    Args:
        output_dir (Path): The directory in which to save the output files.
                           Created if it doesn't exist.
        prompt_files (list[Path]): A list of prompt file paths that were
                                   used for inference.
        results (InferenceResults): An object containing model outputs
                                    and performance metrics.

    Returns:
        None

    Raises:
        SystemExit: If the output directory could not be created.
    """
    # Attempt to create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory at {output_dir}: {e}")
        exit(1)

    # Write inference outputs
    for file, result in zip(prompt_files, results.outputs):
        with open(output_dir / f"{file.stem}.txt", "w") as f:
            f.write(result)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = output_dir / f"performance_metrics_{timestamp}.txt"

    # Write performance metrics
    with open(metrics_filename, "w") as f:
        f.write(f"Results for inference with model: {model}\n")
        f.write(f"Inference Time: {results.inference_time:.4f}s\n")
        f.write(f"TTFT: {results.ttft:.4f}s\n")
        f.write(f"Inference Time w/o TTFT: {results.inference_time - results.ttft:.4f}s\n")
        f.write(f"Number of Output Tokens Generated: {results.output_token_count} tokens\n")
        f.write(f"Throughput: {(results.output_token_count / results.inference_time):.4f}tok/s\n")
        f.write("\n== Per-Prompt Performance Metrics ==\n")
        for i, latencies in enumerate(results.token_latencies):
            min_itl = min(latencies)
            max_itl = max(latencies)
            avg_itl = sum(latencies) / len(latencies)
            f.write(
                f"Prompt {i} ITL (min, max, avg): {min_itl:.4f}s, {max_itl:.4f}s, {avg_itl:.4f}s\n"
            )

    print(f"Saved results to {output_dir}")


def run_inference(
    client: OpenAI,
    model: str,
    tokenizer: PreTrainedTokenizer,
    prompt_files: list[Path],
    max_tokens: int,
    min_tokens: int,
) -> InferenceResults:
    """
    Runs inference using an OpenAI-compatible client on a set of text prompts.

    This function reads prompt files, tokenizes the inputs,
    sends them to the server for streamed completion,
    and calculates performance metrics such as inference time,
    time to first token (TTFT), and inter-token latency (ITL).

    Args:
        client (OpenAI): An instance of the OpenAI client.
        model (str): The model ID to use for inference.
        tokenizer (PreTrainedTokenizer): The tokenizer used to
                                         compute token counts.
        prompt_files (list[Path]): A list of file paths pointing to `.txt`
                                   prompt files.
        max_tokens (int): Maximum number of tokens to generate per prompt.
        min_tokens (int): Minimum number of tokens to generate per prompt.

    Returns:
        InferenceMetrics:
            - outputs (list[str]): Raw list of generated text completions
                                   for each prompt.
            - inference_time (float): Total time taken for
                                      inference (seconds).
            - inference_time_no_ttft (float): Time taken for inference
                                              excluding ttft (seconds).
            - output_token_count (int): Total number of output tokens
                                        generated across all prompts.
            - ttft (float): Time to first token (seconds).
            - itl (float): Inter-token latency (seconds per token).

    Raises:
        Exception: If error occurs during the inference process.
    """
    # Read text from each prompt file
    prompts = [p.read_text() for p in prompt_files]
    # Get token count for each prompt
    for i, (prompt_text, prompt_file) in enumerate(zip(prompts, prompt_files)):
        token_count = len(tokenizer(prompt_text)["input_ids"])
        print(f"Prompt file: {prompt_file.name} | Prompt #{i} token count: {token_count}")

    # Single prompt test run
    print("Starting single prompt test run")
    test_prompt = prompts[0]
    try:
        test_response = client.completions.create(
            model=model,
            prompt=test_prompt,
            max_tokens=max_tokens,
            stream=True,
            temperature=0.0,
            extra_body=dict(min_tokens=min_tokens),
        )

        output = [""]
        for chunk in test_response:
            idx = chunk.choices[0].index
            output[idx] += chunk.choices[0].text
    except Exception as e:
        print("Error during single prompt test run:\n")
        print(e)
        exit(1)
    print("Completed single prompt test run")

    print("Starting inference")
    try:
        # Submit inference payload
        start_time = time.perf_counter()
        response = client.completions.create(
            model=model,
            prompt=prompts,
            max_tokens=max_tokens,
            stream=True,
            temperature=0.0,
            extra_body=dict(min_tokens=min_tokens),
        )

        # Collect streamed tokens
        outputs = [""] * len(prompts)
        ttft = None
        last_token_time: list[
            float  # type: ignore
            | None
        ] = [None] * len(prompts)
        token_latencies: list[list[float]] = [[] for _ in prompts]
        for chunk in response:
            idx = chunk.choices[0].index
            now = time.perf_counter()

            # Record the TTFT
            if ttft is None:
                ttft = now - start_time

            # Record subsequent ITLs per prompt
            if last_token_time[idx] is not None:
                token_latencies[idx].append(now - last_token_time[idx])

            # Update last‐seen and accumulate text
            last_token_time[idx] = now
            outputs[idx] += chunk.choices[0].text
        end_time = time.perf_counter()
        print("Inference complete")

        # Calculate results
        inference_time = end_time - start_time
        output_token_count = sum(len(tokenizer(output)["input_ids"]) for output in outputs)

    except Exception as e:
        print("Error during inference:\n")
        print(e)
        exit(1)

    return InferenceResults(
        outputs,
        inference_time,
        output_token_count,
        ttft,  # type: ignore
        token_latencies,
    )


def main():
    # Collect command line arguments
    args = parse_args()
    tokenizer_dir = args.tokenizer_dir
    port = args.port
    prompt_dir = args.prompt_dir
    max_tokens = args.max_tokens
    min_tokens = args.min_tokens
    output_dir = args.output_dir

    client = create_client("EMPTY", f"http://localhost:{port}/v1")

    # If a server connection is made
    if test_server_connection(client, "/models/"):
        # Prepare model and prompts
        prompt_list = process_input_prompts(prompt_dir)
        tokenizer = get_tokenizer(tokenizer_dir)
        model = get_model_from_server(client)

        # Inference step
        results = run_inference(client, model, tokenizer, prompt_list, max_tokens, min_tokens)

        # Print results
        for file, result in zip(prompt_list, results.outputs):
            print(f"\n== Output for {file.name} ==\n{result}\n")
        print("\n== Inference Performance Metrics ==")
        print(f"Inference Time: {results.inference_time:.4f}s")
        print(f"TTFT: {results.ttft:.4f}s")
        print(f"Inference Time w/o TTFT: {results.inference_time - results.ttft:.4f}s")
        print(f"Number of Output Tokens Generated: {results.output_token_count} tokens")
        print(f"Throughput: {(results.output_token_count / results.inference_time):.4f}tok/s")
        print("\n== Per-Prompt Performance Metrics ==")
        for i, latencies in enumerate(results.token_latencies):
            min_itl = min(latencies)
            max_itl = max(latencies)
            avg_itl = sum(latencies) / len(latencies)
            print(f"Prompt {i} ITL (min, max, avg): {min_itl:.4f}s, {max_itl:.4f}s, {avg_itl:.4f}s")

        # Optionally save results
        if output_dir:
            save_results(output_dir, prompt_list, model, results)
    else:
        print("Server connection failed")
        exit(1)


if __name__ == "__main__":
    main()
