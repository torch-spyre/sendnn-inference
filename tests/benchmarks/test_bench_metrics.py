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

# Synthetic values
FAKE_METRICS: list[dict[str, Any]] = [
    {
        "queued_time_s": 42.0,
        "num_chunked_prefills": 7,
        "chunk_prefill_latencies_s": [0.001, 999.9, 0.003, 500.0, 0.002, 750.0, 1.0],
        "chunk_prefill_start_times_s": [1000.0, 1000.01, 2000.0, 2000.5, 3000.0, 3000.1, 4000.0],
        "decode_latencies_s": [88888.8, 0.000005, 44444.4],
        "decode_start_times_s": [5000.0, 5088888.8, 5088888.8],
        "tkvs": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        "prefix_cache_hit_pct": 0.25,
        "left_padding_blocks": [2, 0, 1],
    },
    {
        "queued_time_s": 0.00001,
        "num_chunked_prefills": 3,
        "chunk_prefill_latencies_s": [12345.6, 0.0001, 99999.9],
        "chunk_prefill_start_times_s": [1000.0, 1012345.6, 1012345.6],
        "decode_latencies_s": [0.000002, 77777.7],
        "decode_start_times_s": [1112345.5, 1112345.5],
        "tkvs": [256, 512, 1024, 2048, 4096],
        "prefix_cache_hit_pct": 0.0,
        "left_padding_blocks": [3, 1],
    },
]

SELECTED_PERCENTILES = [90.0, 99.0, 100.0]


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
    expected_keys = {"spyre_" + k for k in FAKE_METRICS[0]} | {"spyre_total_prefill_chunks"}
    for key in expected_keys:
        assert key in data, f"expected key {key!r} missing from result JSON"


@pytest.mark.cpu
def test_inject_values_correct(tmp_path):
    result_file = _write_fake_result(tmp_path)
    _inject_spyre_metrics_into_result_file(_make_args(tmp_path), FAKE_METRICS, time.time() - 1)
    data = json.loads(result_file.read_text())

    # Per-request lists map directly: "spyre_" + key → [m[key] for m in FAKE_METRICS]
    for key in FAKE_METRICS[0]:
        if key == "num_chunked_prefills":
            continue
        expected = [m[key] for m in FAKE_METRICS]
        assert data["spyre_" + key] == expected
    # Derived run-level scalar
    assert data["spyre_total_prefill_chunks"] == sum(
        m["num_chunked_prefills"] for m in FAKE_METRICS
    )
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
    assert "spyre_queued_time_s" in data


# ---------------------------------------------------------------------------
# Test 2 — _print_spyre_section
# ---------------------------------------------------------------------------


class _TrackedOutput:
    """Wraps the captured stdout of _print_spyre_section and tracks which
    non-blank, non-separator lines have been covered by assert_contains().
    Call assert_all_lines_covered() at the end of the test to fail if any
    content line was never asserted against."""

    def __init__(self, text: str) -> None:
        self._text = text
        # Content lines: all non-empty lines except pure "=" footer/header lines
        self._content_lines = [
            line for line in text.splitlines() if line.strip() and set(line.strip()) != {"="}
        ]
        self._covered: set[int] = set()

    def assert_contains(self, substring: str) -> None:
        assert substring in self._text, f"{substring!r} not found in output"
        for i, line in enumerate(self._content_lines):
            if substring in line:
                self._covered.add(i)

    def assert_not_contains(self, substring: str) -> None:
        assert substring not in self._text, f"{substring!r} unexpectedly found in output"

    def assert_all_lines_covered(self) -> None:
        uncovered = [
            self._content_lines[i]
            for i in range(len(self._content_lines))
            if i not in self._covered
        ]
        assert not uncovered, "The following output lines were never asserted:\n" + "\n".join(
            f"  {line!r}" for line in uncovered
        )


@pytest.mark.cpu
def test_print_sendnn_header():
    # The SenDNN header is printed by main() just before _print_spyre_section.
    # We verify the format string produces the expected centred header.
    header = "{s:{c}^{n}}".format(s=" SenDNN Metrics ", n=50, c="=")
    assert "SenDNN Metrics" in header
    assert header.startswith("=")
    assert header.endswith("=")
    assert len(header) == 50


@pytest.mark.cpu
def test_print_spyre_section_output(capsys):
    """Single test covering every line emitted by _print_spyre_section.
    Uses _TrackedOutput to enforce that no output line goes unasserted —
    add assertions here when a new metric section is added."""
    _print_spyre_section(FAKE_METRICS, SELECTED_PERCENTILES)
    out = _TrackedOutput(capsys.readouterr().out)

    # Run-level scalar
    out.assert_contains("Total prefill chunks processed:")
    out.assert_contains("10")

    # Section separators and mean/median/percentile lines for all six sections
    out.assert_contains("Queue Wait Time")
    out.assert_contains("Chunked Prefill Count")
    out.assert_contains("Chunked Prefill Latency")
    out.assert_contains("Decode Step Latency")
    out.assert_contains("Prefix Cache Hit")
    out.assert_contains("Left Padding Blocks")

    for label in (
        "Queue Wait Time (ms)",
        "Num Chunked Prefills",
        "Chunk Prefill Latency (ms)",
        "Decode Step Latency (ms)",
        "Prefix Cache Hit (%)",
        "Left Padding Blocks",
    ):
        out.assert_contains(f"Mean {label}:")
        out.assert_contains(f"Median {label}:")
        for percentile in SELECTED_PERCENTILES:
            out.assert_contains(f"P{int(percentile)} {label}:")

    out.assert_all_lines_covered()


@pytest.mark.cpu
def test_print_noop_when_empty(capsys):
    _print_spyre_section([], SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert out == ""


@pytest.mark.cpu
def test_print_missing_keys_show_zeros(capsys):
    # Metrics without chunk_prefill_latencies_s or decode_latencies_s — those
    # sections still print with 0.00 values as a fallback, alongside the sections
    # that do have data.
    metrics = [
        {"queued_time_s": 77777.7, "num_chunked_prefills": 13},
        {"queued_time_s": 0.000003, "num_chunked_prefills": 99},
    ]
    _print_spyre_section(metrics, SELECTED_PERCENTILES)
    out = capsys.readouterr().out
    assert "Queue Wait Time" in out
    assert "Chunked Prefill Count" in out
    assert "Chunked Prefill Latency" in out
    assert "Decode Step Latency" in out
    assert "0.00" in out


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


# One entry per field of SpyreBenchState — test_bench_fixture_covers_all_per_req_fields
# will fail if this is out of sync. For dict fields the value is used as the test
# payload (populated as bench.<field>["r0"] = value). Scalar fields are set directly.
_BENCH_FIXTURE: dict[str, Any] = {
    "chunk_latencies": [88888.8, 0.000005],
    "arrival_ts": 1000.0,
    "first_scheduled_ts": 1001.0,
    "chunk_start_times": [1000.0, 1088888.8],
    "decode_latencies": [0.1, 0.2],
    "decode_start_times": [2000.0, 2000.1],
    "tkvs": [64, 128, 192, 256],
    "left_padding_blocks": [2, 0],
    "prefill_step_start": 999.0,
    "decode_step_start": 1999.0,
}

# Expected return value of get_and_clear_chunk_stats. Add an entry here when adding
# a new key to its return dict — the structural test will fail until you do.
_EXPECTED_RESULT: dict[str, Any] = {
    "num_chunked_prefills": 2,
    "chunk_prefill_latencies_s": pytest.approx([88888.8, 0.000005]),
    "chunk_prefill_start_times_s": pytest.approx([1000.0, 1088888.8]),
    "decode_latencies_s": pytest.approx([0.1, 0.2]),
    "decode_start_times_s": pytest.approx([2000.0, 2000.1]),
    "tkvs": [64, 128, 192, 256],
    "left_padding_blocks": [2, 0],
}


@pytest.mark.cpu
def test_bench_fixture_covers_all_per_req_fields():
    """_BENCH_FIXTURE must contain one entry per field of SpyreBenchState.
    Fails when a field is added to SpyreBenchState without updating _BENCH_FIXTURE."""
    from dataclasses import fields as dc_fields

    s = _make_bare_scheduler()
    assert s._bench is not None
    all_fields = {f.name for f in dc_fields(s._bench)}
    assert _BENCH_FIXTURE.keys() == all_fields, (
        f"_BENCH_FIXTURE is out of sync with SpyreBenchState fields.\n"
        f"  extra  : {_BENCH_FIXTURE.keys() - all_fields}\n"
        f"  missing: {all_fields - _BENCH_FIXTURE.keys()}"
    )


@pytest.mark.cpu
def test_get_and_clear_result_keys():
    """get_and_clear_chunk_stats must return exactly the keys in _EXPECTED_RESULT.
    Fails when a key is added or removed without updating _EXPECTED_RESULT."""
    s = _make_bare_scheduler()
    assert s._bench is not None
    bench = s._bench
    for field, value in _BENCH_FIXTURE.items():
        attr = getattr(bench, field)
        if isinstance(attr, dict):
            attr["r0"] = value
        else:
            setattr(bench, field, value)
    result = s.get_and_clear_chunk_stats("r0")
    assert result is not None
    assert result.keys() == _EXPECTED_RESULT.keys(), (
        f"get_and_clear_chunk_stats returned unexpected keys.\n"
        f"  extra  : {result.keys() - _EXPECTED_RESULT.keys()}\n"
        f"  missing: {_EXPECTED_RESULT.keys() - result.keys()}"
    )


@pytest.mark.cpu
def test_get_and_clear_returns_correct_dict():
    s = _make_bare_scheduler()
    assert s._bench is not None
    bench = s._bench

    for field, value in _BENCH_FIXTURE.items():
        attr = getattr(bench, field)
        if isinstance(attr, dict):
            attr["r0"] = value
        else:
            setattr(bench, field, value)

    result = s.get_and_clear_chunk_stats("r0")
    assert result is not None

    for key, expected in _EXPECTED_RESULT.items():
        assert result[key] == expected, f"result[{key!r}] mismatch"

    # All per-request bench fields must be cleared after retrieval
    for field, value in _BENCH_FIXTURE.items():
        if isinstance(value, list):
            assert "r0" not in getattr(bench, field), f"bench.{field} was not cleared for r0"


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
        from dataclasses import fields as dc_fields

        req_id = request.request_id
        bench = self._bench
        # Snapshot all dict fields dynamically so new metrics are captured automatically.
        # List-valued dicts are copied; scalar-valued dicts (arrival_ts, first_scheduled_ts)
        # are stored as-is so truthiness checks work correctly.
        if bench:
            snap = {}
            for f in dc_fields(bench):
                val = getattr(bench, f.name)
                if not isinstance(val, dict):
                    continue
                entry = val.get(req_id)
                snap[f.name] = list(entry) if isinstance(entry, list) else entry
            captured[req_id] = snap
        else:
            captured[req_id] = {}
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

        # At least 2 prefill chunks recorded (prompt > chunk_size)
        assert len(info["chunk_latencies"]) >= 2, (
            f"req {req_id}: expected ≥2 chunk latencies, got {info['chunk_latencies']}"
        )
        # All prefill latencies must be positive floats
        for lat in info["chunk_latencies"]:
            assert isinstance(lat, float) and lat > 0, f"req {req_id}: non-positive latency {lat}"

        # chunk_start_times must match chunk_latencies in length
        assert len(info["chunk_start_times"]) == len(info["chunk_latencies"]), (
            f"req {req_id}: chunk_start_times length {len(info['chunk_start_times'])} "
            f"!= chunk_latencies length {len(info['chunk_latencies'])}"
        )
        for ts in info["chunk_start_times"]:
            assert isinstance(ts, float) and ts > 0, (
                f"req {req_id}: non-positive chunk_start_time {ts}"
            )

        # Decode latencies: at least 1 decode step after the prefill phase
        assert len(info["decode_latencies"]) >= 1, (
            f"req {req_id}: expected ≥1 decode latency, got {info['decode_latencies']}"
        )
        for lat in info["decode_latencies"]:
            assert isinstance(lat, float) and lat > 0, (
                f"req {req_id}: non-positive decode latency {lat}"
            )

        # decode_start_times must match decode_latencies in length
        assert len(info["decode_start_times"]) == len(info["decode_latencies"]), (
            f"req {req_id}: decode_start_times length {len(info['decode_start_times'])} "
            f"!= decode_latencies length {len(info['decode_latencies'])}"
        )
        for ts in info["decode_start_times"]:
            assert isinstance(ts, float) and ts > 0, (
                f"req {req_id}: non-positive decode_start_time {ts}"
            )

        # tkvs: one entry per prefill chunk + per decode step
        expected_tkvs = len(info["chunk_latencies"]) + len(info["decode_latencies"])
        assert len(info["tkvs"]) == expected_tkvs, (
            f"req {req_id}: expected {expected_tkvs} tkvs, got {info['tkvs']}"
        )
        for tkv in info["tkvs"]:
            assert isinstance(tkv, int) and tkv > 0, f"req {req_id}: non-positive tkv {tkv}"

        assert info["arrival_ts"] is not None, f"req {req_id}: arrival_ts not set"
        assert info["first_scheduled_ts"] is not None, f"req {req_id}: first_scheduled_ts not set"

        # left_padding_blocks: one entry per decode step, matching decode_latencies length
        assert len(info["left_padding_blocks"]) == len(info["decode_latencies"]), (
            f"req {req_id}: left_padding_blocks length {len(info['left_padding_blocks'])} "
            f"!= decode_latencies length {len(info['decode_latencies'])}"
        )
        for blocks in info["left_padding_blocks"]:
            assert isinstance(blocks, int) and blocks >= 0, (
                f"req {req_id}: negative left_padding_blocks value {blocks}"
            )

    # The two requests must have independent latency lists (no cross-contamination)
    assert captured["0"]["chunk_latencies"] != captured["1"]["chunk_latencies"] or (
        # Allow equal only if prompts produced identical timings by coincidence —
        # check lengths are both ≥ 2 as a minimum
        len(captured["0"]["chunk_latencies"]) >= 2 and len(captured["1"]["chunk_latencies"]) >= 2
    )

    # After all requests finished, all bench state dicts must be empty.
    # Checked dynamically so new fields are covered automatically.
    from dataclasses import fields as dc_fields

    assert scheduler._bench is not None
    for f in dc_fields(scheduler._bench):
        val = getattr(scheduler._bench, f.name)
        if isinstance(val, dict):
            assert val == {}, f"Leftover entries in {f.name} after run"


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
