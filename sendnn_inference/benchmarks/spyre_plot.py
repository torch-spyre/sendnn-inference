# SPDX-License-Identifier: Apache-2.0
"""Detailed per-request Gantt-chart timeline plot for sendnn-bench serve.

Generates an HTML file showing, for each request:
  - Waiting time before first prefill (ttft - sum of prefill latencies)
  - Each individual prefill chunk as a separate segment
  - Waiting gaps between segments (absorbed if < 10% of segment duration)
  - Each individual decode step (with TKV in hover)
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Color scheme
_COLOR_WAITING = "#636EFA"  # blue-purple — waiting (time before first prefill)
_COLOR_PREFILL = "#FF0092"  # pink — all prefill chunks
_COLOR_DECODE_FAST = "#109618"  # green — decode ITL below lower threshold (or all decodes)
_COLOR_DECODE_MID = "#FF7F0E"  # orange — decode ITL between thresholds (vLLM colors)
_COLOR_DECODE_SLOW = "#D62728"  # red — decode ITL above upper threshold
_GAP_ABSORPTION_THRESHOLD = 0.10  # absorb gap if < 10% of current segment duration


def _tostr(sec: float) -> str:
    """Convert elapsed seconds to HH:MM:SS.mmm (same format as vLLM's plot.py)."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _decode_labels(thresholds: list[float] | None) -> tuple[str, str, str]:
    """Return (fast_label, mid_label, slow_label) matching vLLM's timeline format."""
    if not thresholds:
        return ("Decode", "", "")
    lo_ms = int(thresholds[0] * 1000)
    hi_ms = int(thresholds[1] * 1000)
    return (
        f"ITL < {lo_ms}ms",
        f"{lo_ms}ms ≤ ITL < {hi_ms}ms",
        f"ITL ≥ {hi_ms}ms",
    )


def _decode_type(lat: float, thresholds: list[float] | None, labels: tuple[str, str, str]) -> str:
    fast, mid, slow = labels
    if not thresholds:
        return fast
    if lat < thresholds[0]:
        return fast
    if lat < thresholds[1]:
        return mid
    return slow


def _build_detailed_segments(
    request: dict[str, Any],
    t0_global: float,
    itl_thresholds: list[float] | None = None,
    decode_labels: tuple[str, str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert one request's timing data into ordered Gantt segments.

    All timestamps are elapsed seconds relative to t0_global (min client start_time
    across all requests), formatted as HH:MM:SS.mmm strings for px.timeline.

    ttft - sum(chunk_prefill_latencies_s) gives the waiting time before the first
    prefill (plus any inter-prefill gaps, which are shown as separate "Waiting" segments).
    All values are client-clock, so there is no cross-clock skew.
    """
    client_start = (request.get("start_time") or 0.0) - t0_global
    latency = request.get("latency")
    prompt_len = request.get("prompt_len")
    output_tokens = request.get("output_tokens")
    req_finish = client_start + latency if latency is not None else None

    prefill_lats = request.get("chunk_prefill_latencies_s") or []
    prefill_starts_abs = request.get("chunk_prefill_start_times_s") or []
    decode_lats = request.get("decode_latencies_s") or []
    decode_starts_abs = request.get("decode_start_times_s") or []
    tkvs = request.get("tkvs") or []

    if not prefill_lats:
        return []

    segments: list[dict[str, Any]] = []
    req_label = request.get("_label", "Req ?")

    common = {
        "request_id": req_label,
        "prompt_tokens": prompt_len,
        "output_tokens": output_tokens,
        "req_start_time": _tostr(client_start),
        "req_finish_time": _tostr(req_finish) if req_finish is not None else "—",
    }

    # --- Waiting (client send → first prefill) ---
    # ttft covers: waiting + all prefill chunks + inter-prefill gaps.
    # Subtracting the sum of prefill latencies leaves waiting + gaps; gaps are then
    # shown separately as "Waiting" segments driven by server timestamps, so this
    # initial bar ends up covering only the pre-first-prefill wait.
    # All values are client-clock, so there is no cross-clock skew.
    ttft = request.get("ttft") or 0.0
    waiting_time_s = max(ttft - sum(prefill_lats), 0.0)
    first_prefill_t = client_start + waiting_time_s
    segments.append(
        {
            **common,
            "start": _tostr(client_start),
            "end": _tostr(first_prefill_t),
            "type": "Waiting",
            "duration": f"{waiting_time_s * 1000:.1f}ms",
            "tkv": "—",
        }
    )

    # --- Prefill chunks ---
    # Use server-side absolute timestamps only to derive inter-chunk gaps
    # (t_start[i+1] - t_start[i] - lat[i] = gap between chunk i and i+1).
    # Cursor advances from first_prefill_t using latencies + derived gaps.
    cursor = first_prefill_t
    prev_end_cursor = cursor  # tracks where previous segment ended
    tkv_idx = 0  # index into all_tkvs (one per prefill, then one per decode)

    for i, lat in enumerate(prefill_lats):
        seg_start = cursor

        # Derive gap from server-side start-time diff when available
        if i > 0 and len(prefill_starts_abs) > i:
            gap = prefill_starts_abs[i] - prefill_starts_abs[i - 1] - prefill_lats[i - 1]
            gap = max(gap, 0.0)
            if gap > lat * _GAP_ABSORPTION_THRESHOLD:
                # Insert explicit waiting segment
                segments.append(
                    {
                        **common,
                        "start": _tostr(prev_end_cursor),
                        "end": _tostr(prev_end_cursor + gap),
                        "type": "Waiting",
                        "duration": f"{gap * 1000:.1f}ms",
                        "tkv": "—",
                    }
                )
                seg_start = prev_end_cursor + gap
            # else: absorb gap — seg_start stays at prev_end_cursor (no waiting bar)
            cursor = seg_start

        tkv = tkvs[tkv_idx] if tkv_idx < len(tkvs) else None
        tkv_idx += 1
        seg_end = seg_start + lat
        segments.append(
            {
                **common,
                "start": _tostr(seg_start),
                "end": _tostr(seg_end),
                "type": "Prefill",
                "duration": f"{lat * 1000:.1f}ms",
                "tkv": str(tkv) if tkv is not None else "—",
            }
        )
        prev_end_cursor = seg_end
        cursor = seg_end

    # --- Transition gap: last prefill → first decode ---
    if decode_lats and prefill_starts_abs and decode_starts_abs:
        gap = decode_starts_abs[0] - prefill_starts_abs[-1] - prefill_lats[-1]
        gap = max(gap, 0.0)
        if gap > decode_lats[0] * _GAP_ABSORPTION_THRESHOLD:
            segments.append(
                {
                    **common,
                    "start": _tostr(prev_end_cursor),
                    "end": _tostr(prev_end_cursor + gap),
                    "type": "Waiting",
                    "duration": f"{gap * 1000:.1f}ms",
                    "tkv": "—",
                }
            )
            cursor = prev_end_cursor + gap
        # else absorb

    # --- Decode steps ---
    for i, lat in enumerate(decode_lats):
        seg_start = cursor

        if i > 0 and len(decode_starts_abs) > i:
            gap = decode_starts_abs[i] - decode_starts_abs[i - 1] - decode_lats[i - 1]
            gap = max(gap, 0.0)
            if gap > lat * _GAP_ABSORPTION_THRESHOLD:
                segments.append(
                    {
                        **common,
                        "start": _tostr(cursor),
                        "end": _tostr(cursor + gap),
                        "type": "Waiting",
                        "duration": f"{gap * 1000:.1f}ms",
                        "tkv": "—",
                    }
                )
                seg_start = cursor + gap

        tkv = tkvs[tkv_idx] if tkv_idx < len(tkvs) else None
        tkv_idx += 1
        seg_end = seg_start + lat
        segments.append(
            {
                **common,
                "start": _tostr(seg_start),
                "end": _tostr(seg_end),
                "type": _decode_type(lat, itl_thresholds, decode_labels or ("Decode", "", "")),
                "duration": f"{lat * 1000:.1f}ms",
                "tkv": str(tkv) if tkv is not None else "—",
            }
        )
        cursor = seg_end

    return segments


def generate_detailed_timeline_plot(
    requests: list[dict[str, Any]],
    output_path: Path,
    itl_thresholds: list[float] | None = None,
) -> None:
    """Build a per-request Gantt-chart HTML and write it to output_path.

    Args:
        itl_thresholds: Two thresholds in seconds [low, high]. Decode steps below
            low are green, between low and high are orange, above high are red.
            When None (default), all decode steps are green.
    """
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.io as pio
    except ImportError as exc:
        logger.warning(
            "Cannot generate detailed timeline plot — missing dependency: %s. "
            "Install with: pip install plotly pandas",
            exc,
        )
        return

    if not requests:
        logger.warning("No request data to plot — skipping detailed timeline.")
        return

    valid_starts = [r["start_time"] for r in requests if r.get("start_time") is not None]
    if not valid_starts:
        logger.warning("No start_time in collected requests — skipping detailed timeline.")
        return
    t0_global = min(valid_starts)

    sorted_requests = sorted(requests, key=lambda r: r.get("start_time") or 0.0)
    for idx, req in enumerate(sorted_requests):
        req["_label"] = f"Req {idx}"

    labels = _decode_labels(itl_thresholds)
    fast_lbl, mid_lbl, slow_lbl = labels

    all_segments: list[dict[str, Any]] = []
    for req in sorted_requests:
        all_segments.extend(_build_detailed_segments(req, t0_global, itl_thresholds, labels))

    if not all_segments:
        logger.warning("No plottable segments found — skipping detailed timeline.")
        return

    df = pd.DataFrame(all_segments)

    color_map: dict[str, str] = {
        "Waiting": _COLOR_WAITING,
        "Prefill": _COLOR_PREFILL,
        fast_lbl: _COLOR_DECODE_FAST,
    }
    category_order = ["Waiting", "Prefill", fast_lbl]
    if itl_thresholds:
        color_map[mid_lbl] = _COLOR_DECODE_MID
        color_map[slow_lbl] = _COLOR_DECODE_SLOW
        category_order += [mid_lbl, slow_lbl]

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="request_id",
        color="type",
        color_discrete_map=color_map,
        category_orders={"type": category_order},
        hover_data=[
            "prompt_tokens",
            "output_tokens",
            "req_start_time",
            "req_finish_time",
            "duration",
            "tkv",
        ],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Type: %{fullData.name}<br>"
            "Duration: %{customdata[4]}<br>"
            "TKV: %{customdata[5]}<br>"
            "Prompt tokens: %{customdata[0]}<br>"
            "Output tokens: %{customdata[1]}<br>"
            "Req start: %{customdata[2]}<br>"
            "Req end: %{customdata[3]}<br>"
            "<extra></extra>"
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time (HH:MM:SS from run start)",
        yaxis_title="Request",
        legend_title_text="Segment type",
    )

    try:
        pio.write_html(fig, str(output_path))
        logger.info("Detailed timeline written to %s", output_path)
        print(f"Detailed timeline plot written to: {output_path}")
    except Exception as exc:
        logger.warning("Failed to write detailed timeline HTML: %s", exc)
