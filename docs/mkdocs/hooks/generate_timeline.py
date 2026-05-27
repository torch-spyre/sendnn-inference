"""Generate Plotly HTML plots showing request timelines with TTFT and ITLs."""

import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_PATH = ROOT_DIR / "docs/mkdocs/data"
OUTPUT_DIR = ROOT_DIR / "docs/assets/plots"

SAVE_OUTPUT = True

# Colors
TTFT_COLOR = "#636EFA"
ITL_SHORT_COLOR = "#109618"  # ITL < 5 unit time (length 1)
ITL_MEDIUM_COLOR = "#FF7F0E"  # 5 unit time ≤ ITL < 10 unit time (length 9)


def load_timeline_data(file_path: str) -> list[dict]:
    """Load timeline data from a JSONL file."""
    with open(file_path, encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    return requests


def create_timeline_figure(requests: list[dict]) -> go.Figure:
    """Create a timeline figure from request data."""
    fig = go.Figure()

    # Track which labels we've added to legend
    legend_added = {"ttft": False, "itl_short": False, "itl_medium": False}

    # Plot each request (reverse order so Request 0 is at top)
    for idx, request in enumerate(reversed(requests)):
        req_id = request["id"]
        arrival_time = request["arrival_time"]
        ttft = request["ttft"]
        itls = request["itls"]

        y_pos = f"Request {req_id}"

        # Plot TTFT bar starting at arrival_time
        hovertemplate = f"Request {req_id}<br>TTFT: {ttft}<br>Start: {arrival_time}<extra></extra>"
        fig.add_trace(
            go.Bar(
                x=[ttft],
                y=[y_pos],
                orientation="h",
                marker=dict(color=TTFT_COLOR),
                base=arrival_time,
                name="TTFT",
                legendgroup="ttft",
                showlegend=not legend_added["ttft"],
                hovertemplate=hovertemplate,
            )
        )
        legend_added["ttft"] = True

        # Current position after TTFT
        current_pos = arrival_time + ttft

        # Plot ITL bars
        for itl in itls:
            if itl == 1:
                color = ITL_SHORT_COLOR
                label = "ITL < 5 unit time"
                legend_key = "itl_short"
            elif itl == 9:
                color = ITL_MEDIUM_COLOR
                label = "5 unit time ≤ ITL < 10 unit time"
                legend_key = "itl_medium"
            else:
                color = "gray"
                label = "Other ITL"
                legend_key = "other"

            hovertemplate = f"Request {req_id}<br>ITL: {itl}<br>Start: {current_pos}<extra></extra>"
            fig.add_trace(
                go.Bar(
                    x=[itl],
                    y=[y_pos],
                    orientation="h",
                    marker=dict(color=color),
                    base=current_pos,
                    name=label,
                    legendgroup=legend_key,
                    showlegend=not legend_added.get(legend_key, False),
                    hovertemplate=hovertemplate,
                )
            )
            legend_added[legend_key] = True
            current_pos += itl

    # Update layout
    fig.update_layout(
        title="Request Timeline with TTFT and ITLs",
        xaxis_title="Time (arbitrary unit time)",
        yaxis_title="Request ID",
        barmode="stack",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        hovermode="closest",
    )

    return fig


def generate_timeline_plot(file_path: str, show_figure: bool = True) -> None:
    """Generate timeline plot from JSONL data.

    Args:
        file_path: Path to JSONL file containing timeline data.
        show_figure: Whether to display the figure interactively. Default True.
    """
    requests = load_timeline_data(file_path)
    fig = create_timeline_figure(requests)

    if SAVE_OUTPUT:
        # Save to OUTPUT_DIR with the base filename
        base_name = Path(file_path).stem
        output_path = OUTPUT_DIR / f"{base_name}.html"
        pio.write_html(fig, str(output_path), auto_play=False)

    if show_figure:
        fig.show()


def on_pre_build(config):
    """MkDocs hook that runs before the build."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots for all timeline JSON files in the data directory
    json_files = list(DATA_PATH.glob("timeline_*.json"))

    for json_file in json_files:
        try:
            print(f"Generating timeline plot for: {json_file.name}")
            generate_timeline_plot(file_path=str(json_file), show_figure=False)
        except Exception as e:
            print(f"Error generating timeline plot for {json_file.name}: {e}")


def main():
    """Run the plot generation manually."""
    # Example: generate plot for the admission constraints timeline
    timeline_file = DATA_PATH / "timeline_admission_constraints.json"
    if timeline_file.exists():
        generate_timeline_plot(file_path=str(timeline_file))
    else:
        print(f"Timeline file not found: {timeline_file}")


if __name__ == "__main__":
    main()

# Made with Bob
