"""Generate Plotly HTML plots showing cache demonstration with two rows."""

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


def load_plot_data(file_path: str) -> tuple[dict, dict, list[dict]]:
    """Load metadata and per-step scheduling data from a JSONL file."""
    with open(file_path, encoding="utf-8") as f:
        metadata = json.loads(f.readline())
        requests = json.loads(f.readline())
        steps = [json.loads(line) for line in f]
    return metadata, requests, steps


def build_prefilling_plot_data(step: dict, chunk_size: int) -> dict[str, Any]:
    """Build prefilling-queue bar values for the single in-progress prefill request."""
    prefilling = step.get("prefilling")
    if prefilling is None:
        return {
            "req_id": [" "],
            "left_padding_x": [0.0],
            "cached_token_x": [0.0],
            "right_padding_x": [0.0],
            "inner_padding_x": [0.0],
            "chunks_done_x": [0.0],
            "chunk_current_x": [0.0],
            "chunks_remaining_x": [0.0],
            "label": [""],
            "chunk_info": "",
            "active_chunk_info": "",
            "has_active_chunk": False,
            "prompt_len": None,
        }

    chunks_total = prefilling["chunks_total"]
    chunks_done = prefilling["chunks_done"]
    chunk_prompt_token_start = prefilling["chunk_prompt_token_start"]
    chunk_prompt_token_end = prefilling["chunk_prompt_token_end"]
    left_padding = prefilling["left_padding"]
    right_padding = prefilling["right_padding"]
    inner_padding = prefilling["inner_padding"]
    cached_tokens = prefilling["cached_tokens"]

    # Check if the prompt range for the current chunk corresponds to an already-completed chunk
    expected_chunk_start = chunks_done * chunk_size - left_padding
    is_chunk_already_done = chunks_done > 0 and chunk_prompt_token_start < expected_chunk_start

    if is_chunk_already_done:
        # The chunk shown is already done, no current prefilling tokens
        tokens_done = chunk_prompt_token_end - cached_tokens
        tokens_current = 0
        tokens_remaining = prefilling["prompt_len"] - chunk_prompt_token_end
        active_chunk_info = ""
        active_chunk_start = 0.0
        active_chunk_end = 0.0
        has_active_chunk = False
    else:
        # Normal case: chunk is being processed
        tokens_done = max(0, chunk_prompt_token_start - cached_tokens - inner_padding)
        tokens_current = chunk_prompt_token_end - chunk_prompt_token_start
        tokens_remaining = prefilling["prompt_len"] - chunk_prompt_token_end
        active_chunk_info = f" (chunk {chunks_done + 1} / {chunks_total})"
        active_chunk_start = (
            float(left_padding) + float(cached_tokens) + float(inner_padding) + float(tokens_done)
        )
        active_chunk_end = (
            float(left_padding)
            + float(cached_tokens)
            + float(inner_padding)
            + float(tokens_done)
            + float(chunk_size)
        )
        has_active_chunk = tokens_current > 0

    tokens_remaining = max(tokens_remaining, 0)
    tokens_done = max(tokens_done, 0)

    left_padding = float(left_padding)
    right_padding = float(right_padding)
    inner_padding = float(inner_padding)

    chunk_label = f"chunk {chunks_done + 1}/{chunks_total}"
    req_label = f"{prefilling['id']}"
    chunk_info = f" (chunk {chunks_done + 1} / {chunks_total})"

    return {
        "req_id": [req_label],
        "left_padding_x": [left_padding],
        "cached_token_x": [cached_tokens],
        "right_padding_x": [right_padding],
        "inner_padding_x": [inner_padding],
        "chunks_done_x": [float(tokens_done)],
        "chunk_current_x": [float(tokens_current)],
        "chunks_remaining_x": [float(tokens_remaining)],
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
                xref="x2",
                yref="y2",
                x0=0,
                y0=0.45,
                x1=0,
                y1=1.55,
                fillcolor="rgba(0, 0, 0, 0)",
                line=dict(color="rgba(0, 0, 0, 0)", width=3),
                layer="above",
            )
        ]

    return [
        dict(
            type="rect",
            xref="x2",
            yref="y2",
            x0=(prefilling_data["active_chunk_start"] // chunk_size) * chunk_size,
            y0=0.45,
            x1=(prefilling_data["active_chunk_end"] // chunk_size) * chunk_size,
            y1=1.55,
            fillcolor="rgba(255, 0, 0, 0.10)",
            line=dict(color="rgba(255, 0, 0, 0.95)", width=2),
            layer="above",
        )
    ]


def build_chunk_grid_shapes(max_model_len: int, chunk_size: int) -> list[dict]:
    """Build bold vertical grid lines for chunk size multiples (bottom plot only)."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        return []

    chunk_grid_shapes = []
    for x_pos in range(0, max_model_len + 1, chunk_size):
        # Add line for bottom subplot (row 2) only
        chunk_grid_shapes.append(
            dict(
                type="line",
                xref="x2",
                yref="y2 domain",
                x0=x_pos,
                y0=0,
                x1=x_pos,
                y1=1,
                line=dict(color="rgba(0, 0, 0, 0.5)", width=2.5, dash="dot"),
                layer="above",
            )
        )
    return chunk_grid_shapes


def create_figure() -> go.Figure:
    """Create the base subplot layout with two rows: Requests and Prefill."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Requests", "Prefill"),
        vertical_spacing=0.18,
        row_heights=[0.35, 0.65],
    )
    fig.update_annotations(font_size=12)
    return fig


def add_initial_traces(
    fig: go.Figure,
    requests_data: dict[str, Any],
    prefilling_data: dict[str, Any],
) -> None:
    """Add initial bars to both subplots."""
    # --- Top plot: Static requests (row 1) ---
    # Request 2 (new request) - now first
    fig.add_trace(
        go.Bar(
            x=[requests_data["common_prompt_tokens"] + requests_data["prefix_output_tokens"]],
            y=[requests_data["req2_id"]],
            marker_color="#4CAF50",
            orientation="h",
            name="Prompt Prefix",
            legendgroup="requests",
            legendgrouptitle_text="Requests",
            legendrank=6,
            width=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["req2_diff_prompt"]],
            y=[requests_data["req2_id"]],
            marker_color="#FFC107",
            orientation="h",
            name="Rest Prompt 2",
            legendgroup="requests",
            legendrank=3,
            width=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[0.0],
            y=[requests_data["req2_id"]],
            marker_color="#0000CC",
            orientation="h",
            name="Output Prefix",
            legendgroup="requests",
            showlegend=False,  # Same as req1, don't duplicate in legend
            width=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["req2_max_output"]],
            y=[requests_data["req2_id"]],
            marker_color="#99ccff",
            orientation="h",
            name="Max Output Tokens",
            legendgroup="requests",
            legendrank=1,
            width=0.7,
        ),
        row=1,
        col=1,
    )

    # Request 1 (in cache) - now second
    fig.add_trace(
        go.Bar(
            x=[requests_data["common_prompt_tokens"]],
            y=[requests_data["req1_id"]],
            marker_color="#4CAF50",
            orientation="h",
            name="Prompt Prefix",
            legendgroup="requests",
            showlegend=False,  # Same as req2, don't duplicate in legend
            width=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["req1_diff_prompt"]],
            y=[requests_data["req1_id"]],
            marker_color="#FF6666",
            orientation="h",
            name="Rest Prompt 1",
            legendgroup="requests",
            legendrank=4,
            width=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["prefix_output_tokens"]],
            y=[requests_data["req1_id"]],
            marker_color="#0000CC",
            orientation="h",
            name="Output Prefix",
            legendgroup="requests",
            legendrank=5,
            width=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["req1_output"]],
            y=[requests_data["req1_id"]],
            marker_color="#6666FF",
            orientation="h",
            name="Output Rest",
            legendgroup="requests",
            legendrank=2,
            width=0.7,
        ),
        row=1,
        col=1,
    )

    # --- Bottom plot: Dynamic prefilling (row 2) ---
    # Add reference bar (thin) with fixed values
    fig.add_trace(
        go.Bar(
            x=[prefilling_data["left_padding_x"][0]],
            y=[" "],
            marker_color="#A9A9A9",
            orientation="h",
            name="dummy_reference_padding",
            legendgroup="prefill",
            showlegend=False,
            width=0.2,  # Thin bar
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["common_prompt_tokens"] + requests_data["prefix_output_tokens"]],
            y=[" "],
            marker_color="#4CAF50",
            orientation="h",
            name="dummy_reference_prefix",
            legendgroup="prefill",
            showlegend=False,
            width=0.2,  # Thin bar
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[requests_data["req2_diff_prompt"]],
            y=[" "],
            marker_color="#FFC107",
            orientation="h",
            name="dummy_reference_rest",
            legendgroup="prefill",
            showlegend=False,
            width=0.2,  # Thin bar
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[prefilling_data["right_padding_x"][0]],
            y=[" "],
            marker_color="#A9A9A9",
            orientation="h",
            name="dummy_reference_right_padding",
            legendgroup="prefill",
            showlegend=False,
            width=0.2,  # Thin bar
        ),
        row=2,
        col=1,
    )

    # Main prefilling bars
    fig.add_trace(
        go.Bar(
            x=prefilling_data["left_padding_x"],
            y=prefilling_data["req_id"],
            marker_color="#A9A9A9",
            orientation="h",
            name="Padding",
            legendgroup="prefill",
            legendgrouptitle_text="Prefill",
            legendrank=1,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["cached_token_x"],
            y=prefilling_data["req_id"],
            marker_color="#9C27B0",
            orientation="h",
            name="Cached Blocks",
            legendgroup="prefill",
            legendrank=5,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["inner_padding_x"],
            y=prefilling_data["req_id"],
            marker_color="#A9A9A9",
            orientation="h",
            name="dummy_block_padding2",
            legendgroup="prefill",
            showlegend=False,  # Second padding, don't show in legend
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=prefilling_data["chunks_done_x"],
            y=prefilling_data["req_id"],
            marker_color="#FF0092",
            orientation="h",
            name="Completed Prefill",
            legendgroup="prefill",
            legendrank=4,
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
            name="Currently Prefilling",
            legendgroup="prefill",
            legendrank=3,
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
            name="Remaining Prompt",
            legendgroup="prefill",
            legendrank=2,
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
            name="dummy_block_padding",
            legendgroup="prefill",
            showlegend=False,  # Second padding, don't show in legend
        ),
        row=2,
        col=1,
    )


def create_frame(
    step_index: int,
    requests_data: dict[str, Any],
    prefilling_data: dict[str, Any],
    chunk_size: int,
    max_model_len: int,
) -> go.Frame:
    """Create one animation frame with static top plot and dynamic bottom plot."""
    step_label = f"{step_index}"

    # Add chunk grid lines
    shapes = build_chunk_grid_shapes(max_model_len, chunk_size)

    # Add overlay rectangle for the currently active prefilling chunk
    shapes.extend(build_prefilling_chunk_overlay(prefilling_data, chunk_size))

    # Update the prefilling subplot title with prompt length and active chunk information
    prompt_len = prefilling_data["prompt_len"]
    prompt_len_str = f" (prompt len {prompt_len})" if prompt_len is not None else ""
    prefilling_title = f"Prefill{prompt_len_str}{prefilling_data.get('active_chunk_info', '')}"

    annotations = [
        dict(
            text="Requests",
            xref="paper",
            yref="paper",
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12),
        ),
        dict(
            text=prefilling_title,
            xref="paper",
            yref="paper",
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12),
        ),
    ]

    return go.Frame(
        data=[
            # Top plot (static) - Request 2 (new request) first
            go.Bar(
                x=[requests_data["common_prompt_tokens"] + requests_data["prefix_output_tokens"]],
                y=[requests_data["req2_id"]],
            ),
            go.Bar(x=[requests_data["req2_diff_prompt"]], y=[requests_data["req2_id"]]),
            go.Bar(x=[0.0], y=[requests_data["req2_id"]]),
            go.Bar(x=[requests_data["req2_max_output"]], y=[requests_data["req2_id"]]),
            # Top plot (static) - Request 1 (in cache) second
            go.Bar(x=[requests_data["common_prompt_tokens"]], y=[requests_data["req1_id"]]),
            go.Bar(x=[requests_data["req1_diff_prompt"]], y=[requests_data["req1_id"]]),
            go.Bar(
                x=[requests_data["prefix_output_tokens"]], y=[requests_data["req1_id"]]
            ),  # don't show it
            go.Bar(x=[requests_data["req1_output"]], y=[requests_data["req1_id"]]),
            # Bottom plot (dynamic) - Reference bar (very thin, fixed values)
            go.Bar(x=[prefilling_data["left_padding_x"][0]], y=[" "]),
            go.Bar(
                x=[requests_data["common_prompt_tokens"] + requests_data["prefix_output_tokens"]],
                y=[" "],
            ),
            go.Bar(x=[requests_data["req2_diff_prompt"]], y=[" "]),
            go.Bar(x=[prefilling_data["right_padding_x"][0]], y=[" "]),
            # Bottom plot (dynamic) - Main prefilling bars
            go.Bar(x=prefilling_data["left_padding_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["cached_token_x"], y=prefilling_data["req_id"]),
            go.Bar(x=prefilling_data["inner_padding_x"], y=prefilling_data["req_id"]),
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


def build_frames(
    steps: list[dict], requests_data: dict, chunk_size: int, max_model_len: int
) -> list[go.Frame]:
    """Build animation frames for all scheduling steps."""
    frames = []

    for i, step in enumerate(steps):
        prefilling_data = build_prefilling_plot_data(step, chunk_size)
        frame = create_frame(i, requests_data, prefilling_data, chunk_size, max_model_len)
        frames.append(frame)

    return frames


def configure_figure_layout(
    fig: go.Figure, max_model_len: int, block_size: int, frames: list[go.Frame], metadata: dict
) -> None:
    """Apply axis and animation controls to the figure."""
    chunk_size = metadata["chunk_size"]
    title_text = (
        f"Max Model Len: {max_model_len} | Block Size: {block_size} | Chunk Size: {chunk_size}"
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=14),
            pad=dict(b=0),
        ),
        margin=dict(t=50),
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
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
                "pad": {"r": 10, "t": 80},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": -0.7,
                "yanchor": "bottom",
            }
        ],
        sliders=[
            {
                "yanchor": "bottom",
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
                },
                "len": 0.9,
                "x": 0.1,
                "y": -0.7,
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

    # Configure x-axes for both subplots
    tick_vals = list(range(0, max_model_len + 1, block_size))
    tick_text = [f"<b>{val}</b>" if val % chunk_size == 0 else str(val) for val in tick_vals]

    # Top plot (requests)
    fig.update_xaxes(
        range=[0, max_model_len],
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickfont=dict(size=10),
        row=1,
        col=1,
    )

    # Bottom plot (prefilling)
    fig.update_xaxes(
        range=[0, max_model_len],
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickfont=dict(size=10),
        row=2,
        col=1,
    )

    # Bottom plot y-axis - tighter spacing
    fig.update_yaxes(
        range=[-0.3, 1.8],
        row=2,
        col=1,
    )

    fig.frames = frames


def generate_plots(
    data: dict | None = None, file_path: str | None = None, show_figure: bool = True
) -> None:
    """Generate cache demonstration plots from JSONL data.

    Args:
        data: Unused parameter (kept for compatibility)
        file_path: Path to JSONL file. If None, raises an error.
        show_figure: Whether to display the figure interactively. Default True.
    """
    del data

    if file_path is None:
        raise ValueError("file_path must be provided")

    metadata, requests_data, prefill_steps = load_plot_data(file_path)
    max_model_len = metadata["max_model_len"]
    block_size = metadata["block_size"]
    chunk_size = metadata.get("chunk_size", block_size)

    fig = create_figure()

    # Build initial prefilling data for bottom plot
    step0 = prefill_steps[0]
    initial_prefilling_data = build_prefilling_plot_data(step0, chunk_size)

    add_initial_traces(fig, requests_data, initial_prefilling_data)

    frames = build_frames(prefill_steps, requests_data, chunk_size, max_model_len)

    # Add initial shapes: chunk grid lines + overlay for the active chunk
    initial_shapes = build_chunk_grid_shapes(max_model_len, chunk_size)
    initial_shapes.extend(build_prefilling_chunk_overlay(initial_prefilling_data, chunk_size))

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
        base_name = Path(file_path).stem
        output_path = OUTPUT_DIR / f"{base_name}.html"
        pio.write_html(fig, str(output_path), auto_play=False)

    if show_figure:
        fig.show()


def on_pre_build(config):
    """MkDocs hook that runs before the build."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots for all JSON files in the data directory
    json_files = list(DATA_PATH.glob("prefix_caching_*.json"))

    for json_file in json_files:
        try:
            print(f"Generating cache demo plot for: {json_file.name}")
            generate_plots(file_path=str(json_file), show_figure=False)
        except Exception as e:
            print(f"Error generating cache demo plot for {json_file.name}: {e}")


def main():
    """Run the plot generation manually."""
    generate_plots(file_path=str(DATA_PATH / "prefix_caching_1.json"))


if __name__ == "__main__":
    main()

# Made with Bob
