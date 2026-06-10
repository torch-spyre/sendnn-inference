# SPDX-License-Identifier: Apache-2.0
"""sendnn-bench serve — vllm bench serve extended with Spyre per-request metrics.

Usage:
    sendnn-bench serve --host localhost --port 8000 --model <model> \\
        --dataset-name random --num-prompts 20 --request-rate 2

Env var:
    SENDNN_INFERENCE_BENCH_METRICS_ENABLED=1  (must also be set on the server)
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any

import numpy as np

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
)
from vllm.benchmarks.serve import add_cli_args, main_async

from sendnn_inference.benchmarks.spyre_request_func import async_request_spyre_chat

logger = logging.getLogger(__name__)

_BACKEND_NAME = "spyre-chat"

# Shared accumulator — populated by the wrapper below during the benchmark run.
_spyre_metrics_collected: list[dict[str, Any]] = []


def _make_collecting_func():
    """Return a wrapper around async_request_spyre_chat that accumulates
    custom_metrics_dict into _spyre_metrics_collected."""

    async def _wrapper(
        request_func_input: RequestFuncInput,
        session,
        pbar=None,
    ):
        output = await async_request_spyre_chat(request_func_input, session, pbar)
        if output.success:
            if output.custom_metrics_dict:
                _spyre_metrics_collected.append(output.custom_metrics_dict)
            else:
                logger.warning(
                    "Spyre metrics absent from response — is "
                    "SENDNN_INFERENCE_BENCH_METRICS_ENABLED set on the server?"
                )
        return output

    return _wrapper


def _register_backend() -> None:
    from vllm.benchmarks.lib.endpoint_request_func import OPENAI_COMPATIBLE_BACKENDS

    ASYNC_REQUEST_FUNCS[_BACKEND_NAME] = _make_collecting_func()
    # Register as an OpenAI-compatible backend so that vllm's main_async
    # enables ignore_eos for random datasets and allows sampling parameters.
    if _BACKEND_NAME not in OPENAI_COMPATIBLE_BACKENDS:
        OPENAI_COMPATIBLE_BACKENDS.append(_BACKEND_NAME)


def _build_parser() -> argparse.ArgumentParser:
    """Build an arg parser based on vllm's but with spyre-chat as default backend."""
    parser = argparse.ArgumentParser(
        description="Spyre-extended vllm bench serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Register our backend before add_cli_args so it appears in --backend choices.
    _register_backend()
    add_cli_args(parser)

    # Override the default so --backend doesn't need to be specified explicitly.
    for action in parser._actions:
        if action.dest == "backend":
            action.default = _BACKEND_NAME
            break

    return parser


def _print_spyre_section(
    metrics_list: list[dict[str, Any]],
    selected_percentiles: list[float],
) -> None:
    """Print Spyre-specific metrics in vllm bench serve format."""
    if not metrics_list:
        return

    queue_times_ms = [m["queued_time_s"] * 1000 for m in metrics_list if "queued_time_s" in m]
    num_chunks_list = [
        m["num_chunked_prefills"] for m in metrics_list if "num_chunked_prefills" in m
    ]
    chunk_lats_ms = [
        lat * 1000 for m in metrics_list for lat in m.get("chunk_prefill_latencies_s", [])
    ]
    total_prefill_chunks = sum(num_chunks_list)

    # Scalar summary line (mirrors vllm's plain-count header section)
    print("{:<40} {:<10}".format("Total prefill chunks processed:", total_prefill_chunks))

    def _section(header: str, values: list[float], label: str) -> None:
        if not values:
            return
        arr = np.array(values)
        print("{s:{c}^{n}}".format(s=f" {header} ", n=50, c="-"))
        print("{:<40} {:<10.2f}".format(f"Mean {label}:", float(np.mean(arr))))
        print("{:<40} {:<10.2f}".format(f"Median {label}:", float(np.median(arr))))
        for p in selected_percentiles:
            print(
                "{:<40} {:<10.2f}".format(
                    f"P{int(p) if int(p) == p else p} {label}:",
                    float(np.percentile(arr, p)),
                )
            )

    _section("Queue Wait Time", queue_times_ms, "Queue Wait Time (ms)")
    _section("Chunked Prefill Count", num_chunks_list, "Num Chunked Prefills")
    _section("Chunked Prefill Latency", chunk_lats_ms, "Chunk Prefill Latency (ms)")

    print("=" * 50)


def _inject_spyre_metrics_into_result_file(
    args: Any,
    metrics_list: list[dict[str, Any]],
    run_started_at: float,
) -> None:
    """If vllm wrote a result JSON (--save-result / --append-result / --result-filename),
    find it and inject per-request Spyre metric lists alongside vllm's own per-request
    fields (ttfts, itls, …)."""
    if not metrics_list:
        return
    if not (
        getattr(args, "save_result", False)
        or getattr(args, "append_result", False)
        or getattr(args, "result_filename", None)
    ):
        return

    # Locate the file vllm just wrote by finding the newest .json in the result dir
    # that was modified after we started the run.
    result_dir = getattr(args, "result_dir", None) or "."
    explicit_name = getattr(args, "result_filename", None)

    if explicit_name:
        candidate = (
            explicit_name
            if os.path.isabs(explicit_name)
            else os.path.join(result_dir, explicit_name)
        )
        candidates = [candidate] if os.path.isfile(candidate) else []
    else:
        try:
            candidates = [
                os.path.join(result_dir, f)
                for f in os.listdir(result_dir)
                if f.endswith(".json")
                and os.path.getmtime(os.path.join(result_dir, f)) >= run_started_at
            ]
        except OSError:
            candidates = []

    if not candidates:
        logger.warning("Could not locate vllm result JSON to inject Spyre metrics into.")
        return

    file_path = max(candidates, key=os.path.getmtime)

    try:
        with open(file_path, encoding="utf-8") as fh:
            result = json.load(fh)
    except Exception as exc:
        logger.warning("Failed to read vllm result JSON %s: %s", file_path, exc)
        return

    result["spyre_queue_times_s"] = [
        m["queued_time_s"] for m in metrics_list if "queued_time_s" in m
    ]
    result["spyre_num_chunked_prefills"] = [
        m["num_chunked_prefills"] for m in metrics_list if "num_chunked_prefills" in m
    ]
    result["spyre_chunk_prefill_latencies_s"] = [
        m.get("chunk_prefill_latencies_s", []) for m in metrics_list
    ]
    result["spyre_total_prefill_chunks"] = sum(result["spyre_num_chunked_prefills"])

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        logger.info("Spyre metrics injected into %s", file_path)
    except Exception as exc:
        logger.warning("Failed to write Spyre metrics into result JSON %s: %s", file_path, exc)


def _run_vllm_and_capture_trailing(args: Any) -> tuple[str, str]:
    """Run vllm's main_async, letting stdout/stderr pass through live until the
    closing '=' * 50 line of the metrics table.  Everything written after that
    marker is captured and returned as (stdout_trailing, stderr_trailing)."""
    import io
    import sys

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    done = {"v": False}

    class _StdoutSplitter:
        def write(self, s):
            if not done["v"]:
                if s.strip() == "=" * 50:
                    done["v"] = True
                else:
                    orig_stdout.write(s)
            else:
                stdout_buf.write(s)

        def flush(self):
            orig_stdout.flush()

    class _StderrSplitter:
        def write(self, s):
            if not done["v"]:
                orig_stderr.write(s)
            else:
                stderr_buf.write(s)

        def flush(self):
            if not done["v"]:
                orig_stderr.flush()

    sys.stdout = _StdoutSplitter()
    sys.stderr = _StderrSplitter()
    try:
        asyncio.run(main_async(args))
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return stdout_buf.getvalue(), stderr_buf.getvalue()


def main() -> None:
    import sys

    # Allow `sendnn-bench serve <args>` as an alias (the word "serve" is ignored).
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        argv = argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Force chat endpoint and our backend.
    args.backend = _BACKEND_NAME
    if not hasattr(args, "endpoint") or args.endpoint == "/v1/completions":
        args.endpoint = "/v1/chat/completions"

    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    _spyre_metrics_collected.clear()

    run_started_at = time.time()
    stdout_trailing, stderr_trailing = _run_vllm_and_capture_trailing(args)

    print("{s:{c}^{n}}".format(s=" SenDNN Metrics ", n=50, c="="))
    _print_spyre_section(_spyre_metrics_collected, selected_percentiles)

    _inject_spyre_metrics_into_result_file(args, _spyre_metrics_collected, run_started_at)

    trailing = stdout_trailing + stderr_trailing
    if trailing.strip():
        print(trailing, end="")


if __name__ == "__main__":
    main()
