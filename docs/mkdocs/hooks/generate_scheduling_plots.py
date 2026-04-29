"""Generate Plotly HTML plots from benchmark JSON data."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_PATH = ROOT_DIR / "docs/mkdocs/data/simple_example.json"
OUTPUT_DIR = ROOT_DIR / "docs/assets/plots"

NUM_WAITING_DISPLAYED = 2
DISPLAY_PREFILL_ONLY = False  # enable for better display performance
SAVE_OUTPUT = True


def get_max_tkv_indices(values: list[float]) -> set[int]:
    positive_tkvs = [value for value in values if value > 0]
    if not positive_tkvs:
        return set()
    max_tkv = max(positive_tkvs)
    return {idx for idx, value in enumerate(values) if value == max_tkv}


def load_plot_data(file_path: str) -> tuple[dict, list[dict]]:
    """Load metadata and per-step scheduling data from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        metadata = json.loads(f.readline())
        steps = [json.loads(line) for line in f]
    return metadata, steps


def build_waiting_plot_data(step: dict, num_waiting_displayed: int) -> tuple[list[float], list[float], list[str]]:
    """Build waiting-queue bar values and pad empty display slots."""
    waiting = step["waiting"][:num_waiting_displayed]
    waiting_prompt_x = [item["prompt_len"] for item in waiting]
    waiting_max_tokens_x = [item["max_tokens"] for item in waiting]
    waiting_req_ids = [item["id"] for item in waiting]

    while len(waiting_prompt_x) < num_waiting_displayed:
        placeholder_idx = len(waiting_prompt_x)
        waiting_prompt_x.append(0.0)
        waiting_max_tokens_x.append(0.0)
        waiting_req_ids.append(" " * (placeholder_idx + 1))

    return waiting_prompt_x, waiting_max_tokens_x, waiting_req_ids


def build_prefilling_plot_data(step: dict, block_size: int) -> dict[str, Any]:
    """Build prefilling-queue bar values for the single in-progress prefill request."""
    is_prefill_step = step.get("step_type", "decode") == "prefill"
    prefilling = step.get("prefilling")
    if prefilling is None:
        return {
            "req_id": [" "],
            "chunks_done_x": [0.0],
            "chunk_current_x": [0.0],
            "chunks_remaining_x": [0.0],
            "max_tokens_x": [0.0],
            "right_padding_x": [0.0],
            "label": [""],
            "chunk_info": "",
        }

    chunks_total = prefilling["chunks_total"]
    chunks_done = prefilling["chunks_done"]

    # Tokens already prefilled (completed chunks)
    tokens_done = prefilling["chunk_start"]
    # Tokens in current chunk being processed
    tokens_current = prefilling["chunk_end"] - prefilling["chunk_start"]
    # Tokens not yet reached (remaining chunks after this one)
    tokens_remaining = prefilling["prompt_len"] - prefilling["chunk_end"]
    tokens_remaining = max(tokens_remaining, 0)
    
    # Check if this is the last chunk
    is_last_chunk = (chunks_done + 1) == chunks_total and is_prefill_step
    
    if is_last_chunk:
        # Calculate right-padding to the next block boundary
        prompt_end = prefilling["chunk_end"]
        # Round up to next block boundary
        next_block_boundary = ((prompt_end + block_size - 1) // block_size) * block_size
        right_padding = next_block_boundary - prompt_end
        max_tokens = 0.0
    else:
        right_padding = 0.0
        max_tokens = prefilling["max_tokens"]

    chunk_label = f"chunk {chunks_done + 1}/{chunks_total}"
    req_label = f"{prefilling['id']}"
    chunk_info = f" (chunk {chunks_done + 1} / {chunks_total})"

    return {
        "req_id": [req_label],
        "chunks_done_x": [float(tokens_done)],
        "chunk_current_x": [float(tokens_current)],
        "chunks_remaining_x": [float(tokens_remaining)],
        "max_tokens_x": [float(max_tokens)],
        "right_padding_x": [float(right_padding)],
        "label": [chunk_label],
        "chunk_info": chunk_info,
    }


def build_decoding_plot_data(
    step: dict,
    batch_size: int,
    previous_completed_ids: Optional[set[str]] = None,
    completed_value_includes_reserved: bool = False,
) -> dict[str, Any]:
    """Build decoding-queue stacked-bar values and per-request TKV values."""
    previous_completed_ids = previous_completed_ids or set()
    completed_ids = set(step.get("completed", []))

    decoding_req_ids = []
    padding_x = []
    prompt_x = []
    decoded_x = []
    decoded_actual = []  # Store actual decoded values for annotations
    reserved_x = []
    completed_x = []
    tkv_values = []
    is_completed = []

    for request in step["decoding"]:
        decoding_req_ids.append(request["id"])

        req_tkv = request.get("tkv", 0)
        tkv_values.append(req_tkv)

        padding = request.get("padding", 0)
        prompt = request["prompt_len"]
        decoded = request.get("decoded", 0)
        reserved = request.get("reserved", 0)
        req_id = request["id"]

        if req_id in completed_ids:
            padding_x.append(0.0)
            prompt_x.append(0.0)
            decoded_x.append(0.0)  # Hide the bar for completed requests
            decoded_actual.append(float(decoded))  # Keep actual value for annotation
            reserved_x.append(reserved)
            completed_x.append(req_tkv)
            is_completed.append(True)
        elif req_id in previous_completed_ids:
            tkv_values[-1] = 0
            padding_x.append(padding)
            prompt_x.append(prompt)
            decoded_x.append(decoded)
            decoded_actual.append(float(decoded))
            reserved_x.append(reserved)
            completed_x.append(0.0)
            is_completed.append(False)
        else:
            padding_x.append(padding)
            prompt_x.append(prompt)
            decoded_x.append(decoded)
            decoded_actual.append(float(decoded))
            reserved_x.append(reserved)
            completed_x.append(0.0)
            is_completed.append(False)

    while len(decoding_req_ids) < batch_size:
        placeholder_idx = len(decoding_req_ids)
        padding_x.append(0.0)
        prompt_x.append(0.0)
        decoded_x.append(0.0)
        decoded_actual.append(0.0)
        reserved_x.append(0.0)
        completed_x.append(0.0)
        decoding_req_ids.append(" " * (placeholder_idx + 1))
        tkv_values.append(0)
        is_completed.append(False)

    return {
        "decoding_req_ids": decoding_req_ids,
        "padding_x": padding_x,
        "prompt_x": prompt_x,
        "decoded_x": decoded_x,
        "decoded_actual": decoded_actual,  # Add actual decoded values
        "reserved_x": reserved_x,
        "completed_x": completed_x,
        "tkv_values": tkv_values,
        "is_completed": is_completed,
        "completed_ids": completed_ids,
    }


def build_tkv_overlay(batch_size: int, tkv_values: list[float]) -> tuple[list[dict], list[dict]]:
    """Build per-request TKV line shapes and annotations."""
    shapes = []
    annotations = [dict(), dict(), dict()]  # First three are for subplot titles
    max_tkv_indices = get_max_tkv_indices(tkv_values)

    for idx in range(batch_size):
        tkv_value = tkv_values[idx]

        if tkv_value <= 0:
            shapes.append(
                dict(
                    type="line",
                    xref="x3",
                    yref="y3",
                    x0=0,
                    y0=idx,
                    x1=0,
                    y1=idx,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ))
            annotations.append(
                dict(
                    xref="x3",
                    yref="y3",
                    x=0,
                    y=idx,
                    text="",
                    showarrow=False,
                    font=dict(color="rgba(0,0,0,0)", size=12),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                ))
            continue

        is_max_tkv = idx in max_tkv_indices
        line_color = "Red"
        text_color = "Red"
        text_weight = "bold" if is_max_tkv else "normal"
        line_width = 5 if is_max_tkv else 3

        shapes.append(
            dict(
                type="line",
                xref="x3",
                yref="y3",
                x0=tkv_value,
                y0=idx - 0.4,
                x1=tkv_value,
                y1=idx + 0.4,
                line=dict(color=line_color, width=line_width, dash="dot"),
            ))
        annotations.append(
            dict(
                xref="x3",
                yref="y3",
                x=tkv_value,
                y=idx + 0.5,
                text=f"tkv={tkv_value}",
                showarrow=False,
                font=dict(color=text_color, size=12, weight=text_weight),
                bgcolor="rgba(0,0,0,0)",  # Fully transparent background
                borderwidth=0,
            ))

    return shapes, annotations


def build_generated_tokens_annotations(
    batch_size: int,
    decoded_actual: list[float],
) -> list[dict]:
    """Build annotations showing the number of generated tokens at the end of each decoding line.
    
    Args:
        batch_size: Maximum number of sequences
        decoded_actual: List of actual decoded (generated) token counts for each request
        
    Returns:
        List of annotation dictionaries for generated token counts
    """
    annotations = []
    has_any_generated = False
    
    for idx in range(batch_size):
        generated_tokens = int(decoded_actual[idx])
        
        has_any_generated = True
        
        # Fixed x position on the right side, aligned with the label
        annotations.append(
            dict(
                xref="paper",
                yref="y3",
                x=1.03,
                y=idx,
                xanchor="left",
                yanchor="middle",
                text=f"<b>{str(generated_tokens)}</b>",
                showarrow=False,
                font=dict(color="#00CC96", size=15),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=2,
                borderpad=4,
            ))
    
    # Add a single "Generated Tokens" label annotation if any tokens were generated
    if has_any_generated:
        annotations.append(
            dict(
                xref="paper",
                yref="paper",
                x=1.03,
                y=0.55,
                xanchor="left",
                yanchor="middle",
                text="<b>Generated Tokens</b>",
                showarrow=False,
                font=dict(color="#00CC96", size=14),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#00CC96",
                borderwidth=2,
                borderpad=4,
            ))
    
    return annotations


def build_volume_overlay(batch_size: int, tkv_values: list[float], num_decoding: int) -> tuple[list[dict], list[dict]]:
    """Build volume overlay rectangle and annotation for decoding requests.
    
    Args:
        batch_size: Maximum number of sequences
        tkv_values: List of TKV values for each request
        num_decoding: Number of active decoding requests
        
    Returns:
        Tuple of (shapes, annotations) for the volume overlay
    """
    shapes = []
    annotations = []
    
    # Determine if there are active decoding requests
    if num_decoding == 0:
        max_tkv = 0
        volume = 0
    else:
        # Calculate max_tkv from positive values
        positive_tkvs = [value for value in tkv_values if value > 0]
        if not positive_tkvs:
            max_tkv = 0
            volume = 0
        else:
            max_tkv = max(positive_tkvs)
            volume = max_tkv * num_decoding
    
    # Always create a rectangle (infinitely small when no decoding)
    # This ensures the shape exists in all frames for proper animation
    shapes.append(
        dict(
            type="rect",
            xref="x3",
            yref="y3",
            x0=0,
            y0=-0.5,
            x1=max_tkv if max_tkv > 0 else 0.001,  # Infinitely small when no decoding
            y1=(num_decoding - 0.5) if num_decoding > 0 else -0.499,  # Infinitely small when no decoding
            fillcolor="rgba(255, 165, 0, 0.2)" if volume > 0 else "rgba(255, 165, 0, 0)",  # Transparent when no volume
            line=dict(color="rgba(255, 140, 0, 0.8)" if volume > 0 else "rgba(255, 140, 0, 0)", width=2 if volume > 0 else 0),
            layer="below",
        )
    )
    
    # Always add volume annotation (shows 0 when no decoding)
    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=1.03,
            y=-0.1,
            xanchor="left",
            yanchor="middle",
            text=f"<b>Volume = {volume}</b>",
            showarrow=False,
            font=dict(color="rgba(255, 140, 0, 1)", size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(255, 140, 0, 0.8)",
            borderwidth=2,
            borderpad=4,
        )
    )
    
    return shapes, annotations


def build_inactive_overlay_shapes(is_prefill_active: bool) -> list[dict]:
    """Build semi-transparent overlay rectangles for inactive sections.
    
    Args:
        is_prefill_active: True if prefill is running, False if decode is running
        
    Returns:
        List of shape dictionaries for overlaying inactive sections
    """
    shapes = []
    
    if is_prefill_active:
        # Prefill is active, so overlay the decode section (row 3)
        shapes.append(
            dict(
                type="rect",
                xref="x3 domain",
                yref="y3 domain",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                fillcolor="rgba(128, 128, 128, 0.3)",
                line=dict(width=0),
                layer="above",
            )
        )
    else:
        # Decode is active, so overlay the prefill section (row 2)
        shapes.append(
            dict(
                type="rect",
                xref="x2 domain",
                yref="y2 domain",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                fillcolor="rgba(128, 128, 128, 0.3)",
                line=dict(width=0),
                layer="above",
            )
        )
    
    return shapes


def create_figure(batch_size: int, num_waiting_displayed: int, chunk_size: int) -> go.Figure:
    """Create the base subplot layout with waiting, prefilling, and decoding rows."""
    return make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Waiting Queue", f"Prefilling (chunk size: {chunk_size})", "Decoding"),
        vertical_spacing=0.15,
        column_widths=[1.0],
        row_heights=[
            0.1 * num_waiting_displayed,
            0.15,
            0.2 * batch_size,
        ],
    )


def add_initial_traces(
    fig: go.Figure,
    waiting_data: tuple[list[float], list[float], list[str]],
    prefilling_data: dict[str, Any],
    decoding_data: dict[str, list],
) -> None:
    """Add initial waiting, prefilling, and decoding bars to the figure."""
    waiting_prompt_x, waiting_max_tokens_x, waiting_req_ids = waiting_data

    # --- Waiting queue (row 1) ---
    fig.add_trace(
        go.Bar(
            x=waiting_prompt_x,
            y=waiting_req_ids,
            marker_color="#FF0092",
            orientation="h",
            name="Prompt Tokens",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=waiting_max_tokens_x,
            y=waiting_req_ids,
            marker_color="#99ccff",
            orientation="h",
            name="Max Output Tokens",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Decoding queue (row 3) - Added first to control legend order ---
    fig.add_trace(
        go.Bar(
            x=decoding_data["padding_x"],
            y=decoding_data["decoding_req_ids"],
            marker_color="#A9A9A9",
            orientation="h",
            name="Padding",
            legendgroup="general",
            legendgrouptitle_text="General",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=decoding_data["prompt_x"],
            y=decoding_data["decoding_req_ids"],
            marker_color="#FF0092",
            orientation="h",
            name="Prompt Tokens",
            legendgroup="general",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=decoding_data["decoded_x"],
            y=decoding_data["decoding_req_ids"],
            marker_color="#00CC96",
            orientation="h",
            name="Generated Tokens",
            legendgroup="general",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=decoding_data["completed_x"],
            y=decoding_data["decoding_req_ids"],
            marker_color="#0000cc",
            orientation="h",
            name="Completed Request",
            legendgroup="general",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=decoding_data["reserved_x"],
            y=decoding_data["decoding_req_ids"],
            marker_color="#99ccff",
            orientation="h",
            name="Max Output Tokens",
            legendgroup="general",
        ),
        row=3,
        col=1,
    )

    # --- Prefilling queue (row 2) ---
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunks_done_x"],
            y=prefilling_data["req_id"],
            marker_color="#FFCCAA",
            orientation="h",
            name="Prefilled & Remaining Prompt",
            legendgroup="chunked_prefill",
            legendgrouptitle_text="Chunked Prefill",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunk_current_x"],
            y=prefilling_data["req_id"],
            marker_color="#FF6600",
            orientation="h",
            name="Current Chunk",
            legendgroup="chunked_prefill",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunks_remaining_x"],
            y=prefilling_data["req_id"],
            marker_color="#FFCCAA",
            orientation="h",
            name="Prefilled & Remaining Prompt",
            legendgroup="chunked_prefill",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["max_tokens_x"],
            y=prefilling_data["req_id"],
            marker_color="#99ccff",
            orientation="h",
            name="Max Output Tokens",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["right_padding_x"],
            y=prefilling_data["req_id"],
            marker_color="#A9A9A9",
            orientation="h",
            name="Right Padding",
            legendgroup="chunked_prefill",
            showlegend=False,
        ),
        row=2,
        col=1,
    )


def create_frame(
    step_index: int,
    step_type: str,
    waiting_data: tuple[list[float], list[float], list[str]],
    prefilling_data: dict[str, Any],
    decoding_data: dict[str, list],
    batch_size: int,
    chunk_size: int,
    is_prefill_active: bool,
) -> go.Frame:
    """Create one animation frame."""
    waiting_prompt_x, waiting_max_tokens_x, waiting_req_ids = waiting_data
    shapes, annotations = build_tkv_overlay(
        batch_size=batch_size,
        tkv_values=decoding_data["tkv_values"],
    )

    # Add volume overlay rectangle and annotation
    num_decoding = len([req for req in decoding_data["decoding_req_ids"] if req.strip()])
    volume_shapes, volume_annotations = build_volume_overlay(
        batch_size=batch_size,
        tkv_values=decoding_data["tkv_values"],
        num_decoding=num_decoding,
    )
    shapes.extend(volume_shapes)
    annotations.extend(volume_annotations)

    # Add overlay rectangles to show inactive sections
    overlay_shapes = build_inactive_overlay_shapes(is_prefill_active)
    shapes.extend(overlay_shapes)

    # Add generated tokens annotations
    generated_tokens_annotations = build_generated_tokens_annotations(
        batch_size=batch_size,
        decoded_actual=decoding_data["decoded_actual"],
    )
    annotations.extend(generated_tokens_annotations)

    step_label = f"{step_index}"
    
    # Update the prefilling subplot title with chunk size and chunk information
    prefilling_title = f"Prefilling (chunk size: {chunk_size}){prefilling_data.get('chunk_info', '')}"
    annotations[1] = dict(
        text=prefilling_title,
        xref="paper",
        yref="paper",
        x=0.5,
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=16),
    )

    return go.Frame(
        data=[
            go.Bar(x=waiting_prompt_x, y=waiting_req_ids),
            go.Bar(x=waiting_max_tokens_x, y=waiting_req_ids),
            go.Bar(x=decoding_data["padding_x"], y=decoding_data["decoding_req_ids"]),
            go.Bar(x=decoding_data["prompt_x"], y=decoding_data["decoding_req_ids"]),
            go.Bar(x=decoding_data["decoded_x"], y=decoding_data["decoding_req_ids"]),
            go.Bar(x=decoding_data["completed_x"], y=decoding_data["decoding_req_ids"]),
            go.Bar(x=decoding_data["reserved_x"], y=decoding_data["decoding_req_ids"]),
            go.Bar(x=prefilling_data["chunks_done_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["chunk_current_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["chunks_remaining_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["max_tokens_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["right_padding_x"], y=prefilling_data["req_id"]),
        ],
        layout=go.Layout(
            shapes=shapes,
            annotations=annotations,
        ),
        name=step_label,
    )


def build_frames(steps: list[dict], batch_size: int, num_waiting_displayed: int, display_prefill_only: bool, block_size: int, chunk_size: int) -> list[go.Frame]:
    """Build animation frames for all scheduling steps."""
    frames = []
    previous_frame = None
    previous_completed_ids = set()

    for i, step in enumerate(steps):
        step_type = step.get("step_type", "decode")
        waiting_data = build_waiting_plot_data(step, num_waiting_displayed)
        prefilling_data = build_prefilling_plot_data(step, block_size)
        decoding_data = build_decoding_plot_data(
            step=step,
            batch_size=batch_size,
            previous_completed_ids=previous_completed_ids,
            completed_value_includes_reserved=True,
        )
        # Determine if prefill is active: prefill is active if the step type is "prefill"
        is_prefill_active = step_type == "prefill"
        
        frame = create_frame(i, step_type, waiting_data, prefilling_data, decoding_data, batch_size, chunk_size, is_prefill_active)

        if not display_prefill_only:
            frames.append(frame)
        else:
            is_prefill = len(step["decoding"]) == 0 or step_type == "prefill"
            if is_prefill:
                if previous_frame is not None:
                    frames.append(previous_frame)
                frames.append(frame)
                previous_frame = None
            else:
                previous_frame = frame

        previous_completed_ids = decoding_data["completed_ids"]

    return frames


def configure_figure_layout(fig: go.Figure, batch_size: int, max_model_len: int, block_size: int, tkv_values: list[float], frames: list[go.Frame], metadata: dict, initial_is_prefill_active: bool, initial_num_decoding: int) -> None:
    """Apply axis, TKV overlay, and animation controls to the figure."""
    initial_shapes, initial_annotations = build_tkv_overlay(
        batch_size=batch_size,
        tkv_values=tkv_values,
    )
    
    # # Add initial volume overlay
    initial_volume_shapes, initial_volume_annotations = build_volume_overlay(
        batch_size=batch_size,
        tkv_values=tkv_values,
        num_decoding=initial_num_decoding,
    )
    initial_shapes.extend(initial_volume_shapes)
    initial_annotations.extend(initial_volume_annotations)
    
    # Add initial overlay for inactive sections
    initial_overlay_shapes = build_inactive_overlay_shapes(initial_is_prefill_active)
    initial_shapes.extend(initial_overlay_shapes)

    chunk_size = metadata.get("chunk_size", "N/A")
    max_batch_tkv_limit = metadata.get("max_batch_tkv_limit", "N/A")
    title_text = (
        f"Max Model Len: {max_model_len} | Max Num Seqs: {batch_size} | "
        f"Block Size: {block_size} | Chunk Size: {chunk_size} | Max Volume: {max_batch_tkv_limit}"
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        barmode="stack",
        shapes=initial_shapes,
        annotations=initial_annotations,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 1000, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Stop",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"font": {"size": 16}, "prefix": "step: ", "visible": True, "xanchor": "right"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "steps": [
                    {
                        "method": "animate",
                        "label": frame.name,
                        "args": [
                            [frame.name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame in frames
                ],
            }
        ],
    )

    fig.update_xaxes(range=[0, max_model_len], dtick=block_size, row=1, col=1)
    
    # Configure prefilling x-axis with bold labels at chunk size borders
    if isinstance(chunk_size, int) and chunk_size > 0:
        # Generate tick values at block_size intervals
        tick_vals = list(range(0, max_model_len + 1, block_size))
        # Create tick text with bold formatting at chunk size multiples
        tick_text = [
            f"<b>{val}</b>" if val % chunk_size == 0 else str(val)
            for val in tick_vals
        ]
        fig.update_xaxes(
            range=[0, max_model_len],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            row=2,
            col=1
        )
    else:
        # Fallback if chunk_size is not available
        fig.update_xaxes(range=[0, max_model_len], dtick=block_size, row=2, col=1)
    
    fig.update_xaxes(range=[0, max_model_len], dtick=block_size, row=3, col=1)
    fig.frames = frames


def generate_plots(data: Optional[dict] = None, file_path: Optional[str] = None) -> None:
    """Generate scheduling plots from JSONL data.

    Args:
        data: Unused parameter (kept for compatibility)
        file_path: Path to JSONL file. If None, uses DATA_PATH.
    """
    del data

    if file_path is None:
        file_path = str(DATA_PATH)

    metadata, steps = load_plot_data(file_path)
    max_model_len = metadata["max_model_len"]
    batch_size = metadata["max_num_seqs"]
    block_size = metadata["block_size"]
    chunk_size = metadata.get("chunk_size", "N/A")

    fig = create_figure(batch_size=batch_size, num_waiting_displayed=NUM_WAITING_DISPLAYED, chunk_size=chunk_size)

    step0 = steps[0]
    initial_waiting_data = build_waiting_plot_data(step0, NUM_WAITING_DISPLAYED)
    initial_prefilling_data = build_prefilling_plot_data(step0, block_size)
    initial_decoding_data = build_decoding_plot_data(
        step=step0,
        batch_size=batch_size,
        completed_value_includes_reserved=False,
    )
    add_initial_traces(fig, initial_waiting_data, initial_prefilling_data, initial_decoding_data)

    # Determine if initial step has prefill active
    initial_is_prefill_active = initial_prefilling_data["req_id"][0] != " "

    frames = build_frames(
        steps=steps,
        batch_size=batch_size,
        num_waiting_displayed=NUM_WAITING_DISPLAYED,
        display_prefill_only=DISPLAY_PREFILL_ONLY,
        block_size=block_size,
        chunk_size=chunk_size,
    )
    
    # Calculate initial number of decoding requests
    initial_num_decoding = len([req for req in initial_decoding_data["decoding_req_ids"] if req.strip()])
    
    configure_figure_layout(
        fig=fig,
        batch_size=batch_size,
        max_model_len=max_model_len,
        block_size=block_size,
        tkv_values=initial_decoding_data["tkv_values"],
        frames=frames,
        metadata=metadata,
        initial_is_prefill_active=initial_is_prefill_active,
        initial_num_decoding=initial_num_decoding,
    )

    if SAVE_OUTPUT:
        pio.write_html(fig, os.path.splitext(file_path)[0] + ".html")

    fig.show()


def on_pre_build(_config):
    """MkDocs hook that runs before the build."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Run the plot generation manually."""
    generate_plots({})


if __name__ == "__main__":
    main()
