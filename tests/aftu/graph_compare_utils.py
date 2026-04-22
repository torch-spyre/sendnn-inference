"""Utilities for comparing computation graphs between SenDNN-Inference and AFTU.

This module provides functions to:
- Load and normalize graph files for comparison
- Run AFTU inference and collect generated graphs
- Compare graphs and generate detailed diff reports
- Save normalized graphs for manual inspection

The normalization process removes non-deterministic elements like:
- Memory addresses (ptr: 0x...)
- Object IDs (id: ...)
- Buffer values (values: ...)
- Symbol names (s1, s2, etc. -> S#0, S#1, etc.)

This allows for meaningful comparison of graph structure and operations.
"""

import difflib
import os
import re
import tempfile
from collections.abc import Iterator
from glob import iglob
from os import path
from subprocess import PIPE, STDOUT, CalledProcessError, TimeoutExpired, run

from spyre_util import ModelInfo
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf


def load_graph_to_compare(file_path):
    with open(file_path) as file:
        content = file.read()

    # Replace id: <number> with id: ###
    content = re.sub(r"id: \d+", "id: ###", content)

    # Replace ptr: <pointer> with ptr: xxxx (match any length hex address)
    content = re.sub(r"ptr: 0x[0-9a-fA-F]+", "ptr: xxxx", content)

    # Replace value
    content = re.sub(r"values: ([0-9a-fA-F]{2}\s*)+", "values: $$", content)

    # Regex to find all 's#' patterns surrounded by spaces,
    # or starting with a space and ending with a comma.
    # Examples: ' s1 ', ' s1,', ' s1 s2 '
    matched_symbols = re.findall(r"\s*(s\d+)[\s|,]", content)

    symbols_set = set([m for m in matched_symbols])

    # reindex symbols, considering the sorted indices

    sorted_symbols = sorted(list(symbols_set))
    symbol_map = {i: s for i, s in enumerate(sorted_symbols)}

    for i, s in symbol_map.items():
        content = content.replace(s, f"S#{i}")

    return content


def collect_graph_files(input_dir: str) -> dict[str, tuple[str, str]]:
    # Get G1 graphs.
    # Assumes the 'input_dir' contains 'export_dtcompiler' with the files.

    filepaths = iglob(path.join(input_dir, "export_dtcompiler", "*/*.ops"))

    # Filter out G2 files
    filepaths = [f for f in filepaths if not f.endswith(".g2.ops")]

    # NOTE: f.split("dump")[-1], split the filename by using dump,
    # to get numeric part which is the last one
    filemap = {f.split("dump")[-1]: (f, load_graph_to_compare(f)) for f in filepaths}

    return filemap


def save_normalized_graphs(graph_map: dict[str, tuple[str, str]], output_dir: str) -> str:
    """Save normalized versions of graphs for easier debugging.

    Args:
        graph_map: Dictionary mapping graph keys to (filepath, normalized_content) tuples
        output_dir: Directory to save normalized graphs

    Returns:
        Path to the normalized graphs directory
    """
    normalized_dir = path.join(output_dir, "normalized_graphs")
    os.makedirs(normalized_dir, exist_ok=True)

    for key, (original_path, normalized_content) in graph_map.items():
        # Create a clean filename from the key
        filename = f"graph{key}.ops"
        normalized_path = path.join(normalized_dir, filename)

        with open(normalized_path, "w") as f:
            f.write(normalized_content)

    return normalized_dir


def diff_graph(a_filepath, a_file, b_filepath, b_file) -> Iterator[str]:
    return difflib.unified_diff(
        a_file.split("\n"), b_file.split("\n"), fromfile=a_filepath, tofile=b_filepath
    )


def compare_graphs(
    a_map: dict[str, tuple[str, str]],
    b_map: dict[str, tuple[str, str]],
    a_label: str = "vLLM",
    b_label: str = "AFTU",
) -> tuple[bool, list[str]]:
    """Compare two sets of graphs and return match status with differences.

    Args:
        a_map: First graph map (typically vLLM)
        b_map: Second graph map (typically AFTU)
        a_label: Label for first graph set
        b_label: Label for second graph set

    Returns:
        Tuple of (all_match, differences_list)
    """
    differences = []
    all_match = True

    for k, a_graph in a_map.items():
        a_filename, a_filedata = a_graph
        b_filename, b_filedata = b_map[k]

        diff = list(diff_graph(a_filename, a_filedata, b_filename, b_filedata))
        if diff:
            all_match = False
            diff_summary = [f"\nDifference in graph {k}:"]
            diff_summary.append(f"  {a_label}: {a_filename}")
            diff_summary.append(f"  {b_label}: {b_filename}")

            # Show first 20 lines of diff
            for line in diff[:20]:
                diff_summary.append(f"  {line}")
            if len(diff) > 20:
                diff_summary.append(f"  [...] Omitted {len(diff) - 20} lines")

            differences.append("\n".join(diff_summary))

    return all_match, differences


def run_inference_py_and_get_graphs(
    inference_py_args: list[str],
    extra_env: dict[str, str] | None = None,
    tmpdir: str | None = None,
) -> tuple[dict[str, tuple[str, str]], str]:
    """Run AFTU inference and collect graphs.

    Args:
        inference_py_args: Command line arguments for inference.py
        extra_env: Additional environment variables
        tmpdir: Directory to use for graphs (if None, creates temporary directory)

    Returns:
        Tuple of (graph_map, tmpdir_path)
    """
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="aftu_graphs_")

    env = os.environ.copy()
    env.update({"DEE_DUMP_GRAPHS": "aftu", "TORCH_SENDNN_CACHE_ENABLE": "0"})
    if extra_env:
        env.update(extra_env)

    try:
        run(
            inference_py_args,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            check=True,
            env=env,
            cwd=tmpdir,
            timeout=600,
        )
    except TimeoutExpired as e:
        print("`inference.py` process timeout!")
        if e.stdout:
            print(e.stdout)
        raise e
    except CalledProcessError as e:
        print(f"`inference.py` Process finished with code {e.returncode}")
        if e.stdout:
            print(e.stdout)
        raise e

    aftu_graphs = collect_graph_files(tmpdir)
    return aftu_graphs, tmpdir


def get_model_path(model: ModelInfo):
    if os.path.isdir(model.name):
        return model.name

    # Get location of model from HF cache.
    return download_weights_from_hf(
        model_name_or_path=model.name,
        cache_dir=None,
        allow_patterns=["*.safetensors", "*.bin", "*.pt"],
        revision=model.revision,
    )
