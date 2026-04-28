"""Generate Plotly HTML plots from benchmark JSON data."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent.parent.parent.parent
# DATA_PATH = ROOT_DIR / "docs/mkdocs/data/data.json"
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
    with open(file_path, "r") as f:
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


def build_running_plot_data(
    step: dict,
    batch_size: int,
    previous_completed_ids: Optional[set[str]] = None,
    completed_value_includes_reserved: bool = False,
) -> dict[str, Any]:
    """Build running-queue stacked-bar values and per-request TKV values."""
    previous_completed_ids = previous_completed_ids or set()
    completed_ids = set(step.get("completed", []))

    running_req_ids = []
    padding_x = []
    prompt_x = []
    decoded_x = []
    reserved_x = []
    completed_x = []
    tkv_values = []
    is_completed = []

    for request in step["running"]:
        running_req_ids.append(request["id"])

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
            decoded_x.append(0.0)
            reserved_x.append(reserved)
            completed_x.append(req_tkv)
            is_completed.append(True)
        elif req_id in previous_completed_ids:
            tkv_values[-1] = 0
            padding_x.append(padding)
            prompt_x.append(prompt)
            decoded_x.append(decoded)
            reserved_x.append(reserved)
            completed_x.append(0.0)
            is_completed.append(False)
        else:
            padding_x.append(padding)
            prompt_x.append(prompt)
            decoded_x.append(decoded)
            reserved_x.append(reserved)
            completed_x.append(0.0)
            is_completed.append(False)

    running_req_ids.reverse()
    padding_x.reverse()
    prompt_x.reverse()
    decoded_x.reverse()
    reserved_x.reverse()
    completed_x.reverse()
    tkv_values.reverse()
    is_completed.reverse()

    while len(running_req_ids) < batch_size:
        placeholder_idx = len(running_req_ids)
        padding_x.append(0.0)
        prompt_x.append(0.0)
        decoded_x.append(0.0)
        reserved_x.append(0.0)
        completed_x.append(0.0)
        running_req_ids.append(" " * (placeholder_idx + 1))
        tkv_values.append(0)
        is_completed.append(False)

    return {
        "running_req_ids": running_req_ids,
        "padding_x": padding_x,
        "prompt_x": prompt_x,
        "decoded_x": decoded_x,
        "reserved_x": reserved_x,
        "completed_x": completed_x,
        "tkv_values": tkv_values,
        "is_completed": is_completed,
        "completed_ids": completed_ids,
    }


def build_tkv_overlay(batch_size: int, tkv_values: list[float], hide_zero_values: bool) -> tuple[list[dict], list[dict]]:
    """Build per-request TKV line shapes and annotations."""
    shapes = []
    annotations = [dict(), dict()]  # First two are for subplot titles
    max_tkv_indices = get_max_tkv_indices(tkv_values)

    for idx in range(batch_size):
        tkv_value = tkv_values[idx]
        if not hide_zero_values and tkv_value <= 0:
            continue

        is_hidden = hide_zero_values and tkv_value == 0
        is_max_tkv = idx in max_tkv_indices
        line_color = "rgba(0,0,0,0)" if is_hidden else "Red"
        text_color = "rgba(0,0,0,0)" if is_hidden else "Red"
        text_weight = "bold" if is_max_tkv else "normal"
        line_width = 5 if is_max_tkv else 3

        shapes.append(
            dict(
                type="line",
                xref="x2",
                yref="y2",
                x0=tkv_value,
                y0=idx - 0.4,
                x1=tkv_value,
                y1=idx + 0.4,
                line=dict(color=line_color, width=line_width, dash="dot"),
            ))
        annotations.append(
            dict(
                xref="x2",
                yref="y2",
                x=tkv_value,
                y=idx + 0.5,
                text=f"tkv={tkv_value}",
                showarrow=False,
                font=dict(color=text_color, size=12, weight=text_weight),
            ))

    return shapes, annotations


def create_figure(batch_size: int, num_waiting_displayed: int) -> go.Figure:
    """Create the base subplot layout."""
    return make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Waiting Queue", "Running"),
        vertical_spacing=0.2,
        column_widths=[1.0],
        row_heights=[0.1 * num_waiting_displayed, 0.2 * batch_size],
    )


def add_initial_traces(
    fig: go.Figure,
    waiting_data: tuple[list[float], list[float], list[str]],
    running_data: dict[str, list],
) -> None:
    """Add initial waiting and running bars to the figure."""
    waiting_prompt_x, waiting_max_tokens_x, waiting_req_ids = waiting_data

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

    fig.add_trace(
        go.Bar(
            x=running_data["padding_x"],
            y=running_data["running_req_ids"],
            marker_color="#A9A9A9",
            orientation="h",
            name="Padding",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=running_data["prompt_x"],
            y=running_data["running_req_ids"],
            marker_color="#FF0092",
            orientation="h",
            name="Prompt Tokens",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=running_data["decoded_x"],
            y=running_data["running_req_ids"],
            marker_color="#00CC96",
            orientation="h",
            name="Decoded Tokens",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=running_data["completed_x"],
            y=running_data["running_req_ids"],
            marker_color="#0000cc",
            orientation="h",
            name="Completed",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=running_data["reserved_x"],
            y=running_data["running_req_ids"],
            marker_color="#99ccff",
            orientation="h",
            name="Max Output Tokens",
        ),
        row=2,
        col=1,
    )


def create_frame(
    step_index: int,
    waiting_data: tuple[list[float], list[float], list[str]],
    running_data: dict[str, list],
    batch_size: int,
) -> go.Frame:
    """Create one animation frame."""
    waiting_prompt_x, waiting_max_tokens_x, waiting_req_ids = waiting_data
    shapes, annotations = build_tkv_overlay(
        batch_size=batch_size,
        tkv_values=running_data["tkv_values"],
        hide_zero_values=True,
    )

    return go.Frame(
        data=[
            go.Bar(x=waiting_prompt_x, y=waiting_req_ids),
            go.Bar(x=waiting_max_tokens_x, y=waiting_req_ids),
            go.Bar(x=running_data["padding_x"], y=running_data["running_req_ids"]),
            go.Bar(x=running_data["prompt_x"], y=running_data["running_req_ids"]),
            go.Bar(x=running_data["decoded_x"], y=running_data["running_req_ids"]),
            go.Bar(x=running_data["completed_x"], y=running_data["running_req_ids"]),
            go.Bar(x=running_data["reserved_x"], y=running_data["running_req_ids"]),
        ],
        layout=go.Layout(
            shapes=shapes,
            annotations=annotations,
        ),
        name=str(step_index),
    )


def build_frames(steps: list[dict], batch_size: int, num_waiting_displayed: int, display_prefill_only: bool) -> list[go.Frame]:
    """Build animation frames for all scheduling steps."""
    frames = []
    previous_frame = None
    previous_completed_ids = set()

    for i, step in enumerate(steps):
        waiting_data = build_waiting_plot_data(step, num_waiting_displayed)
        running_data = build_running_plot_data(
            step=step,
            batch_size=batch_size,
            previous_completed_ids=previous_completed_ids,
            completed_value_includes_reserved=True,
        )
        frame = create_frame(i, waiting_data, running_data, batch_size)

        if not display_prefill_only:
            frames.append(frame)
        else:
            is_prefill = len(step["running"]) == 0 or 10 == 1
            if is_prefill:
                if previous_frame is not None:
                    frames.append(previous_frame)
                frames.append(frame)
                previous_frame = None
            else:
                previous_frame = frame

        previous_completed_ids = running_data["completed_ids"]

    return frames


def configure_figure_layout(fig: go.Figure, batch_size: int, max_model_len: int, block_size: int, tkv_values: list[float], frames: list[go.Frame], metadata: dict) -> None:
    """Apply axis, TKV overlay, and animation controls to the figure."""
    initial_shapes, initial_annotations = build_tkv_overlay(
        batch_size=batch_size,
        tkv_values=tkv_values,
        hide_zero_values=False,
    )

    # Create title with run information
    chunk_size = metadata.get("chunk_size", "N/A")
    title_text = (
        # f"Scheduling Visualization<br>"
        f"Max Model Len: {max_model_len} | Max Num Seqs: {batch_size} | "
        f"Block Size: {block_size} | Chunk Size: {chunk_size}"
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
                                "transition": {"duration": 500},
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
                "currentvalue": {"font": {"size": 20}, "prefix": "step: ", "visible": True, "xanchor": "right"},
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
    fig.update_xaxes(range=[0, max_model_len], dtick=block_size, row=2, col=1)
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

    fig = create_figure(batch_size=batch_size, num_waiting_displayed=NUM_WAITING_DISPLAYED)

    step0 = steps[0]
    initial_waiting_data = build_waiting_plot_data(step0, NUM_WAITING_DISPLAYED)
    initial_running_data = build_running_plot_data(
        step=step0,
        batch_size=batch_size,
        completed_value_includes_reserved=False,
    )
    add_initial_traces(fig, initial_waiting_data, initial_running_data)

    frames = build_frames(
        steps=steps,
        batch_size=batch_size,
        num_waiting_displayed=NUM_WAITING_DISPLAYED,
        display_prefill_only=DISPLAY_PREFILL_ONLY,
    )
    configure_figure_layout(
        fig=fig,
        batch_size=batch_size,
        max_model_len=max_model_len,
        block_size=block_size,
        tkv_values=initial_running_data["tkv_values"],
        frames=frames,
        metadata=metadata,
    )

    if SAVE_OUTPUT:
        pio.write_html(fig, os.path.splitext(file_path)[0] + ".html")

    fig.show()


def on_pre_build(config):
    """MkDocs hook that runs before the build."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # print(f"Generating plots from {DATA_PATH}")
    # with open(DATA_PATH) as f:
    #     data = json.load(f)
    # generate_plots(data)


def main():
    """Run the plot generation manually."""
    generate_plots({})


if __name__ == "__main__":
    main()
