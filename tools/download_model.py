#!/usr/bin/env python3
"""Download a model (or a list of models) from HuggingFace into HF_HUB_CACHE.

Single model:

    python3 tools/download_model.py -m <HF-model-id> [-r <git-tag-or-hash>]

Batch download from a YAML config (list of {repo, revision} entries):

    python3 tools/download_model.py --config .github/ci_model_cache.yaml

"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
]


def download(repo: str, revision: str) -> None:
    logging.info("Downloading model '%s' with revision '%s' ...", repo, revision)
    snapshot_download(repo_id=repo, revision=revision, ignore_patterns=IGNORE_PATTERNS)


def download_from_config(config_path: Path) -> None:
    with config_path.open() as f:
        config = yaml.safe_load(f)
    entries = config.get("models", [])
    if not entries:
        logging.error("No models listed in %s", config_path)
        exit(1)

    with ThreadPoolExecutor(max_workers=len(entries)) as pool:
        futures = [pool.submit(download, e["repo"], e.get("revision", "main")) for e in entries]
        for fut in futures:
            fut.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model", help="HuggingFace model ID")
    parser.add_argument(
        "-r", dest="revision", default="main", help="Git hash, tag, or branch (default='main')"
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="YAML file listing {repo, revision} entries to download in parallel",
    )
    args, _extra_args = parser.parse_known_args()

    if args.config:
        download_from_config(Path(args.config))
    elif args.model:
        download(args.model, args.revision)
    else:
        logging.error("Need to provide a HuggingFace model ID or --config file.")
        exit(1)


if __name__ == "__main__":
    main()
