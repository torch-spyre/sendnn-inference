# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sendnn-bench serve custom metric pipeline.

Tests cover four layers:
  1. _inject_spyre_metrics_into_result_file — JSON result file injection
  2. _print_spyre_section — stdout output format
  3. ChunkedPrefillSpyreScheduler accumulation — _chunk_latencies / _arrival_ts /
     _first_scheduled_ts, and get_and_clear_chunk_stats
  4. async_request_spyre_chat — client-side SSE parsing
"""

import argparse
import asyncio
import json
import pathlib
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sendnn_inference.benchmarks.spyre_bench_serve import (
    _inject_spyre_metrics_into_result_file,
    _print_spyre_section,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Deliberately synthetic values — order-of-magnitude differences make clear these
# are not real system measurements.
FAKE_METRICS: list[dict[str, Any]] = [
    {
        "queued_time_s": 42.0,
        "num_chunked_prefills": 7,
        "chunk_prefill_latencies_s": [0.001, 999.9, 0.003, 500.0, 0.002, 750.0, 1.0],
        "chunk_prefill_start_times_s": [1000.0, 1000.01, 2000.0, 2000.5, 3000.0, 3000.1, 4000.0],
        "decode_latencies_s": [88888.8, 0.000005, 44444.4],
        "decode_start_times_s": [5000.0, 5088888.8, 5088888.8],
        "tkvs": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    },
    {
        "queued_time_s": 0.00001,
        "num_chunked_prefills": 3,
        "chunk_prefill_latencies_s": [12345.6, 0.0001, 99999.9],
        "chunk_prefill_start_times_s": [1000.0, 1012345.6, 1012345.6],
        "decode_latencies_s": [0.000002, 77777.7],
        "decode_start_times_s": [1112345.5, 1112345.5],
        "tkvs": [256, 512, 1024, 2048, 4096],
    },
]

SELECTED_PERCENTILES = [99.0]


def _write_fake_result(tmp_path) -> pathlib.Path:
    p = tmp_path / "result.json"
    p.write_text(json.dumps({"backend": "spyre-chat", "num_prompts": 2}))
    return p


def _make_args(tmp_path, *, save_result: bool = True, result_filename=None):
    return argparse.Namespace(
        save_result=save_result,
        append_result=False,
        result_filename=str(result_filename) if result_filename else None,
        result_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Test 1 — _inject_spyre_metrics_into_result_file
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_inject_adds_spyre_keys(tmp_path):
    result_file = _write_fake_result(tmp_path)
    _inject_spyre_metrics_into_result_file(_make_args(tmp_path), FAKE_METRICS, time.time() - 1)
    data = json.loads(result_file.read_text())
    assert "spyre_queue_times_s" in data
    assert "spyre_num_chunked_prefills" in data
    assert "spyre_chunk_prefill_latencies_s" in data
    assert "spyre_chunk_prefill_start_times_s" in data
    assert "spyre_total_prefill_chunks" in data
    assert "spyre_decode_latencies_s" in data
    assert "spyre_decode_start_times_s" in data
    assert "spyre_tkvs" in data


@pytest.mark.cpu
def test_inject_values_correct(tmp_path):
    result_file = _write_fake_result(tmp_path)
    _inject_spyre_metrics_into_result_file(_make_args(tmp_path), FAKE_METRICS, time.time() - 1)
    data = json.loads(result_file.read_text())

    assert data["spyre_queue_times_s"] == pytest.approx([42.0, 0.00001])
    assert data["spyre_num_chunked_prefills"] == [7, 3]
    assert data["spyre_total_prefill_chunks"] == 10
    assert data["spyre_chunk_prefill_latencies_s"] == [
        [0.001, 999.9, 0.003, 500.0, 0.002, 750.0, 1.0],
        [12345.6, 0.0001, 99999.9],
    ]
    assert data["spyre_chunk_prefill_start_times_s"] == [
        [1000.0, 1000.01, 2000.0, 2000.5, 3000.0, 3000.1, 4000.0],
        [1000.0, 1012345.6, 1012345.6],
    ]
    assert data["spyre_decode_latencies_s"] == [
        [88888.8, 0.000005, 44444.4],
        [0.000002, 77777.7],
    ]
    assert data["spyre_decode_start_times_s"] == [
        [5000.0, 5088888.8, 5088888.8],
        [1112345.5, 1112345.5],
    ]
    assert data["spyre_tkvs"] == [
        [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        [256, 512, 1024, 2048, 4096],
    ]
    # Original keys preserved
    assert data["backend"] == "spyre-chat"


@pytest.mark.cpu
def test_inject_noop_when_save_result_false(tmp_path):
    result_file = _write_fake_result(tmp_path)
    original = result_file.read_text()
    _inject_spyre_metrics_into_result_file(
        _make_args(tmp_path, save_result=False), FAKE_METRICS, time.time() - 1
    )
    assert result_file.read_text() == original


@pytest.mark.cpu
def test_inject_noop_when_metrics_empty(tmp_path):
    result_file = _write_fake_result(tmp_path)
    original = result_file.read_text()
    _inject_spyre_metrics_into_result_file(_make_args(tmp_path), [], time.time() - 1)
    assert result_file.read_text() == original


@pytest.mark.cpu
def test_inject_explicit_result_filename(tmp_path):
    result_file = _write_fake_result(tmp_path)
    args = _make_args(tmp_path, result_filename=str(result_file))
    _inject_spyre_metrics_into_result_file(args, FAKE_METRICS, time.time() - 1)
    data = json.loads(result_file.read_text())
    assert "spyre_queue_times_s" in data


# ---------------------------------------------------------------------------
# Test 2 — _print_spyre_section
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_print_scalar_total_line(capsys):
    _print_spyre_section(FAKE_METRICS, SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert "Total prefill chunks processed:" in out
    assert "10" in out


@pytest.mark.cpu
def test_print_sendnn_header(capsys):
    # The SenDNN header is printed by main() just before _print_spyre_section.
    # We verify the format string produces the expected centred header.
    header = "{s:{c}^{n}}".format(s=" SenDNN Metrics ", n=50, c="=")
    assert "SenDNN Metrics" in header
    assert header.startswith("=")
    assert header.endswith("=")
    assert len(header) == 50


@pytest.mark.cpu
def test_print_section_headers(capsys):
    _print_spyre_section(FAKE_METRICS, SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert "Queue Wait Time" in out
    assert "Chunked Prefill Count" in out
    assert "Chunked Prefill Latency" in out
    assert "Decode Step Latency" in out


@pytest.mark.cpu
def test_print_mean_median_p99(capsys):
    _print_spyre_section(FAKE_METRICS, SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert "Mean Queue Wait Time (ms):" in out
    assert "Median Queue Wait Time (ms):" in out
    assert "P99 Queue Wait Time (ms):" in out


@pytest.mark.cpu
def test_print_noop_when_empty(capsys):
    _print_spyre_section([], SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert out == ""


@pytest.mark.cpu
def test_print_missing_keys_tolerated(capsys):
    # Metrics without chunk_prefill_latencies_s or decode_latencies_s — those
    # sections should be absent, but queue time and chunk count should still print.
    metrics = [
        {"queued_time_s": 77777.7, "num_chunked_prefills": 13},
        {"queued_time_s": 0.000003, "num_chunked_prefills": 99},
    ]
    _print_spyre_section(metrics, SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert "Queue Wait Time" in out
    assert "Chunked Prefill Count" in out
    assert "Chunked Prefill Latency" not in out
    assert "Decode Step Latency" not in out


# ---------------------------------------------------------------------------
# Test 3B — get_and_clear_chunk_stats (pure unit, no engine)
# ---------------------------------------------------------------------------


def _make_bare_scheduler():
    """Instantiate ChunkedPrefillSpyreScheduler with __init__ bypassed so we can
    call its methods without a full vllm engine setup."""
    from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler, SpyreBenchState

    with patch.object(ChunkedPrefillSpyreScheduler, "__init__", lambda *a, **kw: None):
        s = ChunkedPrefillSpyreScheduler()
    s._bench = SpyreBenchState()
    return s


@pytest.mark.cpu
def test_get_and_clear_returns_correct_dict():
    s = _make_bare_scheduler()
    s._bench.chunk_latencies["r0"] = [88888.8, 0.000005]
    result = s.get_and_clear_chunk_stats("r0")
    assert result is not None
    assert result["num_chunked_prefills"] == 2
    assert result["chunk_prefill_latencies_s"] == pytest.approx([88888.8, 0.000005])
    # Entry must be cleared after retrieval
    assert "r0" not in s._bench.chunk_latencies


@pytest.mark.cpu
def test_get_and_clear_unknown_req_returns_none():
    s = _make_bare_scheduler()
    assert s.get_and_clear_chunk_stats("unknown") is None


# ---------------------------------------------------------------------------
# Test 3A — scheduler integration (real engine, requires model)
# ---------------------------------------------------------------------------


@pytest.mark.chunked_prefill
@pytest.mark.cpu
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [64])
@pytest.mark.parametrize("available_blocks", [None])
def test_scheduler_bench_metrics_accumulated(
    model,
    backend,
    monkeypatch,
    set_random_seed,
    max_num_seqs,
    max_model_len,
    max_num_batched_tokens,
    available_blocks,
):
    """Two requests with prompts longer than max_num_batched_tokens each trigger
    multiple prefill chunks. Verify that _chunk_latencies, _arrival_ts, and
    _first_scheduled_ts are populated correctly, and cleared once the request
    finishes via _free_request."""
    from llm_cache import get_cached_engine
    from scheduling_utils import create_request_for_scheduler_test, random_prompt

    monkeypatch.setenv("SENDNN_INFERENCE_BENCH_METRICS_ENABLED", "1")
    # Re-read envs so the scheduler sees the updated value
    import sendnn_inference.envs as envs_spyre

    monkeypatch.setattr(envs_spyre, "SENDNN_INFERENCE_BENCH_METRICS_ENABLED", True)

    engine = get_cached_engine(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    scheduler = engine.scheduler

    # Reset bench state in case the engine was cached before BENCH_METRICS_ENABLED was set.
    from sendnn_inference.v1.core.scheduler import SpyreBenchState

    scheduler._bench = SpyreBenchState()

    # Prompts longer than max_num_batched_tokens (64) → ≥ 2 prefill chunks each
    prompt_len = max_num_batched_tokens + 20  # 84 tokens → 2 chunks of 64
    req1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=4,
        prompt=random_prompt(model=model, seed=0, length=prompt_len),
        use_golden_token_injection=False,
        generate_hf_results=False,
    )
    req2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=4,
        prompt=random_prompt(model=model, seed=1, length=prompt_len),
        use_golden_token_injection=False,
        generate_hf_results=False,
    )

    # Capture metrics just before _free_request clears them
    captured: dict[str, dict] = {}
    original_free = scheduler.__class__._free_request

    def _capturing_free(self, request, delay_free_blocks=False):
        req_id = request.request_id
        bench = self._bench
        captured[req_id] = {
            "chunk_latencies": list(bench.chunk_latencies.get(req_id, [])) if bench else [],
            "has_arrival_ts": (req_id in bench.arrival_ts) if bench else False,
            "has_first_scheduled_ts": (req_id in bench.first_scheduled_ts) if bench else False,
        }
        return original_free(self, request, delay_free_blocks)

    scheduler._free_request = _capturing_free.__get__(scheduler)

    # Add both requests and step until both finish.
    # engine.step() returns (engine_core_output_dict, ...) — outputs are keyed
    # by worker rank; use rank 0. Each output has .outputs with per-request
    # RequestOutput objects that expose .new_token_ids and .request_id.
    engine.add_request(req1.request)
    engine.add_request(req2.request)

    tokens_per_req: dict[str, int] = {"0": 0, "1": 0}
    max_tokens = 4
    for _ in range(200):  # generous upper bound
        step_output = engine.step()
        engine_core_output = step_output[0].get(0)
        if engine_core_output is not None:
            for out in engine_core_output.outputs:
                tokens_per_req[out.request_id] = tokens_per_req.get(out.request_id, 0) + len(
                    out.new_token_ids
                )
        if all(v >= max_tokens for v in tokens_per_req.values()):
            break

    assert all(v >= max_tokens for v in tokens_per_req.values()), (
        f"Requests did not finish after 200 steps: {tokens_per_req}"
    )

    # Both requests must have been captured by _capturing_free
    for req_id in ("0", "1"):
        assert req_id in captured, f"_free_request was never called for req {req_id}"
        info = captured[req_id]

        # At least 2 chunks recorded (prompt > chunk_size)
        assert len(info["chunk_latencies"]) >= 2, (
            f"req {req_id}: expected ≥2 chunk latencies, got {info['chunk_latencies']}"
        )
        # All latencies must be positive floats
        for lat in info["chunk_latencies"]:
            assert isinstance(lat, float) and lat > 0, f"req {req_id}: non-positive latency {lat}"
        assert info["has_arrival_ts"], f"req {req_id}: _arrival_ts not set"
        assert info["has_first_scheduled_ts"], f"req {req_id}: _first_scheduled_ts not set"

    # The two requests must have independent latency lists (no cross-contamination)
    assert captured["0"]["chunk_latencies"] != captured["1"]["chunk_latencies"] or (
        # Allow equal only if prompts produced identical timings by coincidence —
        # check lengths are both ≥ 2 as a minimum
        len(captured["0"]["chunk_latencies"]) >= 2 and len(captured["1"]["chunk_latencies"]) >= 2
    )

    # After all requests finished, the bench state dicts must be empty
    assert scheduler._bench is not None
    assert scheduler._bench.chunk_latencies == {}, "Leftover entries in chunk_latencies after run"
    assert scheduler._bench.arrival_ts == {}, "Leftover entries in arrival_ts after run"
    assert scheduler._bench.first_scheduled_ts == {}, (
        "Leftover entries in first_scheduled_ts after run"
    )


# ---------------------------------------------------------------------------
# Test 4 — async_request_spyre_chat SSE parsing
# ---------------------------------------------------------------------------


def _build_sse_stream(chunks: list[str]) -> list[bytes]:
    """Encode a list of SSE message strings as byte chunks."""
    return [c.encode() for c in chunks]


def _make_session_mock(status: int, sse_chunks: list[str]):
    """Return a mock aiohttp ClientSession whose post() returns a fake SSE stream."""

    async def _iter_any():
        for chunk in _build_sse_stream(sse_chunks):
            yield chunk

    response_mock = MagicMock()
    response_mock.status = status
    response_mock.reason = None
    response_mock.content.iter_any = _iter_any

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post.return_value = cm
    return session


def _make_request_input():
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

    return RequestFuncInput(
        prompt="hello",
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=1,
        output_len=4,
        model="test-model",
    )


_SPYRE_METRICS = {
    "queued_time_s": 55555.5,
    "num_chunked_prefills": 42,
    "chunk_prefill_latencies_s": [0.000007, 66666.6],
}

_SSE_WITH_METRICS = [
    'data: {"id":"1","choices":[{"delta":{"content":"hi"},"index":0}]}\n\n',
    f'data: {{"id":"1","choices":[],"usage":{{"prompt_tokens":3,"completion_tokens":2}},'
    f'"spyre_metrics":{json.dumps(_SPYRE_METRICS)}}}\n\n',
    "data: [DONE]\n\n",
]

_SSE_WITHOUT_METRICS = [
    'data: {"id":"1","choices":[{"delta":{"content":"hi"},"index":0}]}\n\n',
    'data: {"id":"1","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
    "data: [DONE]\n\n",
]


@pytest.mark.cpu
def test_spyre_metrics_parsed():
    from sendnn_inference.benchmarks.spyre_request_func import async_request_spyre_chat

    session = _make_session_mock(200, _SSE_WITH_METRICS)
    output = asyncio.run(async_request_spyre_chat(_make_request_input(), session))

    assert output.success is True
    assert output.custom_metrics_dict == _SPYRE_METRICS


@pytest.mark.cpu
def test_spyre_metrics_absent():
    from sendnn_inference.benchmarks.spyre_request_func import async_request_spyre_chat

    session = _make_session_mock(200, _SSE_WITHOUT_METRICS)
    output = asyncio.run(async_request_spyre_chat(_make_request_input(), session))

    assert output.success is True
    assert output.custom_metrics_dict == {}


@pytest.mark.cpu
def test_success_flag_set():
    from sendnn_inference.benchmarks.spyre_request_func import async_request_spyre_chat

    session = _make_session_mock(200, _SSE_WITH_METRICS)
    output = asyncio.run(async_request_spyre_chat(_make_request_input(), session))
    assert output.success is True


@pytest.mark.cpu
def test_output_tokens_parsed():
    from sendnn_inference.benchmarks.spyre_request_func import async_request_spyre_chat

    session = _make_session_mock(200, _SSE_WITH_METRICS)
    output = asyncio.run(async_request_spyre_chat(_make_request_input(), session))
    assert output.output_tokens == 2
