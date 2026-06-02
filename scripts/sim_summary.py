#!/usr/bin/env python3
"""Summarize a sim_metrics.jsonl in the same format as `vllm bench serve`.

The sim plugin emits one JSON line per finalized request with virtual
timings derived from SENDNN_INFERENCE_SIM_PREFILL_MS / _DECODE_MS. This
script reads that file and prints the familiar serving-benchmark summary.

Usage:
    python sim_summary.py /path/to/sim_metrics.jsonl [--max-concurrency 4]
"""

import argparse
import json
import sys
from pathlib import Path


def percentiles(xs, ps):
    xs = sorted(xs)
    n = len(xs)
    out = []
    for p in ps:
        if n == 0:
            out.append(0.0)
            continue
        idx = min(n - 1, int(round(p * (n - 1) / 100)))
        out.append(xs[idx])
    return out


def peak_concurrency(records):
    events = []
    for r in records:
        events.append((r["virtual_arrival_seconds"], +1))
        events.append((r["virtual_completion_seconds"], -1))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def peak_output_throughput(records, decode_ms, window_s=1.0):
    """Bucket each generated token to its emission instant in virtual time and
    return the max tokens/s over a sliding `window_s`-second window."""
    decode_s = decode_ms / 1000.0
    if decode_s <= 0:
        return 0.0
    emits = []
    for r in records:
        first = r.get("time_to_first_token_seconds", 0) + r["virtual_arrival_seconds"]
        n_decode = r.get("num_decode_steps", 0)
        # First token at `first`; subsequent decode tokens at +decode_s, +2*decode_s, ...
        emits.append(first)
        for i in range(1, n_decode + 1):
            emits.append(first + i * decode_s)
    emits.sort()
    if not emits:
        return 0.0
    # Sliding window
    j = 0
    peak = 0
    for i, t_end in enumerate(emits):
        while emits[j] < t_end - window_s:
            j += 1
        peak = max(peak, i - j + 1)
    return float(peak) / window_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="As reported in the bench summary; informational.")
    ap.add_argument("--decode-ms", type=float, default=None,
                    help="Override (default: median per-token from records).")
    args = ap.parse_args()

    records = []
    with args.path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Drop warmup records that aren't from the bench run (heuristic: keep all
    # if no bench prefix is detected; otherwise filter to bench requests).
    bench_prefixed = [r for r in records if r["request_id"].startswith("cmpl-bench")]
    if bench_prefixed:
        records = bench_prefixed

    if not records:
        print("No records.", file=sys.stderr)
        sys.exit(1)

    n = len(records)
    duration = max(r["virtual_completion_seconds"] for r in records) - min(
        r["virtual_arrival_seconds"] for r in records
    )
    total_input = sum(r["num_prompt_tokens"] for r in records)
    total_output = sum(r["num_generation_tokens"] for r in records)

    ttfts_ms = [r["time_to_first_token_seconds"] * 1000 for r in records]
    tpots_ms = [r["mean_time_per_output_token_seconds"] * 1000 for r in records]
    # Per-token ITL across all (request, token-pair) intervals
    itls_ms: list[float] = []
    for r in records:
        for x in r.get("inter_token_latencies_seconds", []):
            itls_ms.append(x * 1000)
    if not itls_ms:
        itls_ms = list(tpots_ms)  # fallback: per-request mean
    e2es_ms = [r["e2e_latency_seconds"] * 1000 for r in records]

    decode_ms = args.decode_ms
    if decode_ms is None:
        # Take any non-zero per-request decode duration; they should all be the same in v1 sim
        for r in records:
            if r["num_decode_steps"] > 0:
                decode_ms = (r["decode_time_seconds"] / r["num_decode_steps"]) * 1000
                break
        decode_ms = decode_ms or 0.0

    peak_conc = peak_concurrency(records)
    peak_out_tput = peak_output_throughput(records, decode_ms)

    def line(label, val, fmt="<10.2f"):
        print(f"{label:<41s}{format(val, fmt)}")

    print("============ Sim Benchmark Result =============")
    line("Successful requests:", n, fmt="<10d")
    line("Failed requests:", 0, fmt="<10d")
    if args.max_concurrency is not None:
        line("Maximum request concurrency:", args.max_concurrency, fmt="<10d")
    line("Benchmark duration (s):", duration)
    line("Total input tokens:", total_input, fmt="<10d")
    line("Total generated tokens:", total_output, fmt="<10d")
    line("Request throughput (req/s):", n / duration)
    line("Output token throughput (tok/s):", total_output / duration)
    line("Peak output token throughput (tok/s):", peak_out_tput)
    line("Peak concurrent requests:", float(peak_conc))
    line("Total token throughput (tok/s):", (total_input + total_output) / duration)

    def block(title, xs):
        print(title)
        ps = percentiles(xs, [50, 99, 100])
        line("Mean " + title.split()[-1] + " (ms):", sum(xs) / len(xs))
        line("Median " + title.split()[-1] + " (ms):", ps[0])
        line("P99 " + title.split()[-1] + " (ms):", ps[1])
        line("P100 " + title.split()[-1] + " (ms):", ps[2])

    print("---------------Time to First Token----------------")
    line("Mean TTFT (ms):", sum(ttfts_ms) / n)
    p = percentiles(ttfts_ms, [50, 99, 100])
    line("Median TTFT (ms):", p[0])
    line("P99 TTFT (ms):", p[1])
    line("P100 TTFT (ms):", p[2])
    print("-----Time per Output Token (excl. 1st token)------")
    line("Mean TPOT (ms):", sum(tpots_ms) / n)
    p = percentiles(tpots_ms, [50, 99, 100])
    line("Median TPOT (ms):", p[0])
    line("P99 TPOT (ms):", p[1])
    line("P100 TPOT (ms):", p[2])
    print("---------------Inter-token Latency----------------")
    line("Mean ITL (ms):", sum(itls_ms) / max(len(itls_ms), 1))
    p = percentiles(itls_ms, [50, 99, 100])
    line("Median ITL (ms):", p[0])
    line("P99 ITL (ms):", p[1])
    line("P100 ITL (ms):", p[2])
    print("----------------End-to-end Latency----------------")
    line("Mean E2EL (ms):", sum(e2es_ms) / n)
    p = percentiles(e2es_ms, [50, 99, 100])
    line("Median E2EL (ms):", p[0])
    line("P99 E2EL (ms):", p[1])
    line("P100 E2EL (ms):", p[2])
    print("==================================================")


if __name__ == "__main__":
    main()
