"""Generate Plotly HTML plots showing only prefilling from benchmark JSON data."""

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_PATH = ROOT_DIR / "docs/mkdocs/data"
OUTPUT_DIR = ROOT_DIR / "docs/assets/plots"

SAVE_OUTPUT = True


def load_plot_data(file_path: str) -> tuple[dict, list[dict]]:
    """Load metadata and per-step scheduling data from a JSONL file."""
    with open(file_path, encoding="utf-8") as f:
        metadata = json.loads(f.readline())
        steps = [json.loads(line) for line in f]
    return metadata, steps


def build_prefilling_plot_data(step: dict, chunk_size) -> dict[str, Any]:
    """Build prefilling-queue bar values for the single in-progress prefill request."""
    prefilling = step.get("prefilling")
    if prefilling is None:
        return {
            "req_id": [" "],
            "left_padding_x": [0.0],
            "chunks_done_x": [0.0],
            "chunk_current_x": [0.0],
            "chunks_remaining_x": [0.0],
            "right_padding_x": [0.0],
            "label": [""],
            "chunk_info": "",
            "active_chunk_info": "",
            "active_chunk_start": 0.0,
            "active_chunk_end": 0.0,
            "has_active_chunk": False,
            "prompt_len": None,
        }

    chunks_total = prefilling["chunks_total"]
    chunks_done = prefilling["chunks_done"]
    chunk_prompt_token_start = prefilling["chunk_prompt_token_start"]
    chunk_prompt_token_end = prefilling["chunk_prompt_token_end"]
    left_padding = prefilling["left_padding"]

    # Check if the prompt range for the current chunk corresponds to an already-completed chunk
    # This happens when chunks_done > 0 and current prompt range hasn't advanced yet
    # In this case, there's no "Currently Prefilling"
    expected_chunk_start = chunks_done * chunk_size - left_padding
    is_chunk_already_done = chunks_done > 0 and chunk_prompt_token_start < expected_chunk_start

    if is_chunk_already_done:
        # The chunk shown is already done, no current prefilling tokens
        tokens_done = chunk_prompt_token_end
        tokens_current = 0
        tokens_remaining = prefilling["prompt_len"] - chunk_prompt_token_end
        active_chunk_info = ""
        active_chunk_start = 0.0
        active_chunk_end = 0.0
        has_active_chunk = False
    else:
        # Normal case: chunk is being processed
        tokens_done = chunk_prompt_token_start
        tokens_current = chunk_prompt_token_end - chunk_prompt_token_start
        tokens_remaining = prefilling["prompt_len"] - chunk_prompt_token_end
        active_chunk_info = f" (chunk {chunks_done + 1} / {chunks_total})"
        active_chunk_start = float(prefilling.get("left_padding", 0)) + float(
            chunk_prompt_token_start
        )
        active_chunk_end = float(prefilling.get("left_padding", 0)) + float(
            chunk_prompt_token_start + chunk_size
        )
        has_active_chunk = tokens_current > 0

    tokens_remaining = max(tokens_remaining, 0)

    left_padding = float(prefilling.get("left_padding", 0))
    right_padding = float(prefilling.get("right_padding", 0))

    chunk_label = f"chunk {chunks_done + 1}/{chunks_total}"
    req_label = f"{prefilling['id']}"
    chunk_info = f" (chunk {chunks_done + 1} / {chunks_total})"

    return {
        "req_id": [req_label],
        "left_padding_x": [left_padding],
        "chunks_done_x": [float(tokens_done)],
        "chunk_current_x": [float(tokens_current)],
        "chunks_remaining_x": [float(tokens_remaining)],
        "right_padding_x": [right_padding],
        "label": [chunk_label],
        "chunk_info": chunk_info,
        "active_chunk_info": active_chunk_info,
        "active_chunk_start": active_chunk_start,
        "active_chunk_end": active_chunk_end,
        "has_active_chunk": has_active_chunk,
        "prompt_len": prefilling["prompt_len"],
    }


def build_prefilling_chunk_overlay(prefilling_data: dict[str, Any], chunk_size: int) -> list[dict]:
    """Build an overlay rectangle around the currently active prefilling chunk."""
    if not prefilling_data.get("has_active_chunk", False):
        return [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                y0=-0.45,
                x1=0,
                y1=0.45,
                fillcolor="rgba(0, 0, 0, 0)",
                line=dict(color="rgba(0, 0, 0, 0)", width=3),
                layer="above",
            )
        ]

    return [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=(prefilling_data["active_chunk_start"] // chunk_size) * chunk_size,
            y0=-0.45,
            x1=(prefilling_data["active_chunk_end"] // chunk_size) * chunk_size,
            y1=0.45,
            fillcolor="rgba(255, 0, 0, 0.10)",
            line=dict(color="rgba(255, 0, 0, 0.95)", width=2),
            layer="above",
        )
    ]


def create_figure() -> go.Figure:
    """Create the base subplot layout with only prefilling row."""
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Prefilling",),
        vertical_spacing=0.15,
        column_widths=[1.0],
        row_heights=[1.0],
    )
    return fig


def add_initial_traces(
    fig: go.Figure,
    prefilling_data: dict[str, Any],
) -> None:
    """Add initial prefilling bars to the figure."""
    # --- Prefilling queue (row 1) - Keep original bar order, control legend with legendrank ---
    fig.add_trace(
        go.Bar(
            x=prefilling_data["left_padding_x"],
            y=prefilling_data["req_id"],
            marker_color="#A9A9A9",
            orientation="h",
            name="Padding",
            legendrank=1001,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunks_done_x"],
            y=prefilling_data["req_id"],
            marker_color="#FF0092",
            orientation="h",
            name="Completed Prefill",
            legendrank=1004,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunk_current_x"],
            y=prefilling_data["req_id"],
            marker_color="#FF6600",
            orientation="h",
            name="Currently Prefilling",
            legendrank=1003,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunks_remaining_x"],
            y=prefilling_data["req_id"],
            marker_color="#FFCCAA",
            orientation="h",
            name="Remaining Prompt",
            legendrank=1002,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["right_padding_x"],
            y=prefilling_data["req_id"],
            marker_color="#A9A9A9",
            orientation="h",
            name="Padding",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def create_frame(
    step_index: int,
    prefilling_data: dict[str, Any],
    chunk_size: int,
) -> go.Frame:
    """Create one animation frame."""
    step_label = f"{step_index}"

    # Add overlay rectangle for the currently active prefilling chunk
    shapes = build_prefilling_chunk_overlay(prefilling_data, chunk_size)

    # Update the prefilling subplot title with prompt length and active chunk information
    prompt_len = prefilling_data.get("prompt_len")
    prompt_len_str = f" (prompt len {prompt_len})" if prompt_len is not None else ""
    prefilling_title = f"Prefilling{prompt_len_str}{prefilling_data.get('active_chunk_info', '')}"
    annotations = [
        dict(
            text=prefilling_title,
            xref="paper",
            yref="paper",
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12),
        )
    ]

    return go.Frame(
        data=[
            go.Bar(x=prefilling_data["left_padding_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["chunks_done_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["chunk_current_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["chunks_remaining_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["right_padding_x"], y=prefilling_data["req_id"]),
        ],
        layout=go.Layout(
            shapes=shapes,
            annotations=annotations,
        ),
        name=step_label,
    )


def build_frames(steps: list[dict], chunk_size) -> list[go.Frame]:
    """Build animation frames for all scheduling steps."""
    frames = []

    for i, step in enumerate(steps):
        prefilling_data = build_prefilling_plot_data(step, chunk_size)
        frame = create_frame(i, prefilling_data, chunk_size)
        frames.append(frame)

    return frames


def configure_figure_layout(
    fig: go.Figure, max_model_len: int, block_size: int, frames: list[go.Frame], metadata: dict
) -> None:
    """Apply axis and animation controls to the figure."""
    chunk_size = metadata.get("chunk_size", "N/A")
    title_text = (
        f"Max Model Len: {max_model_len} | Block Size: {block_size} | Chunk Size: {chunk_size}"
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,  # Moved legend further down to avoid x-axis labels
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        # Prevent auto-play on page load
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
                                "frame": {"duration": 750, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Stop",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
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
                "currentvalue": {
                    "font": {"size": 10},
                    "prefix": "step: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {
                    "b": 10,
                    "t": 80,
                },  # Increased top padding to provide space for x-axis labels
                "len": 0.9,
                "x": 0.1,
                "font": {"size": 10},
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

    # Configure prefilling x-axis with bold labels at chunk size borders
    if isinstance(chunk_size, int) and chunk_size > 0:
        # Generate tick values at block_size intervals
        tick_vals = list(range(0, max_model_len + 1, block_size))
        # Create tick text with bold formatting at chunk size multiples
        tick_text = [f"<b>{val}</b>" if val % chunk_size == 0 else str(val) for val in tick_vals]
        fig.update_xaxes(
            range=[0, max_model_len],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10),
            row=1,
            col=1,
        )
    else:
        # Fallback if chunk_size is not available
        fig.update_xaxes(
            range=[0, max_model_len], dtick=block_size, tickfont=dict(size=10), row=1, col=1
        )

    fig.frames = frames


def generate_plots(
    data: dict | None = None, file_path: str | None = None, show_figure: bool = True
) -> None:
    """Generate prefill-only scheduling plots from JSONL data.

    Args:
        data: Unused parameter (kept for compatibility)
        file_path: Path to JSONL file. If None, raises an error.
        show_figure: Whether to display the figure interactively. Default True.
    """
    del data

    if file_path is None:
        raise ValueError("file_path must be provided")

    metadata, steps = load_plot_data(file_path)
    max_model_len = metadata["max_model_len"]
    block_size = metadata["block_size"]
    chunk_size = metadata.get("chunk_size", block_size)

    fig = create_figure()

    step0 = steps[0]
    initial_prefilling_data = build_prefilling_plot_data(step0, chunk_size)
    add_initial_traces(fig, initial_prefilling_data)

    frames = build_frames(steps=steps, chunk_size=chunk_size)

    # Add initial overlay shapes for the active chunk
    initial_shapes = build_prefilling_chunk_overlay(initial_prefilling_data, chunk_size)

    configure_figure_layout(
        fig=fig,
        max_model_len=max_model_len,
        block_size=block_size,
        frames=frames,
        metadata=metadata,
    )

    # Update layout with initial shapes
    fig.update_layout(shapes=initial_shapes)

    if SAVE_OUTPUT:
        # Save to OUTPUT_DIR with the base filename
        base_name = Path(file_path).stem
        output_path = OUTPUT_DIR / f"{base_name}.html"
        # Disable auto-play to prevent animation from starting automatically
        pio.write_html(fig, str(output_path), auto_play=False)

    if show_figure:
        fig.show()


def on_pre_build(config):
    """MkDocs hook that runs before the build."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots for all JSON files in the data directory
    json_files = list(DATA_PATH.glob("prefill_*.json"))

    for json_file in json_files:
        try:
            print(f"Generating prefill-only plot for: {json_file.name}")
            generate_plots(file_path=str(json_file), show_figure=False)
        except Exception as e:
            print(f"Error generating prefill-only plot for {json_file.name}: {e}")


def main():
    """Run the plot generation manually."""
    generate_plots(file_path=DATA_PATH / "simple_example.json")


if __name__ == "__main__":
    main()

# Made with Bob
