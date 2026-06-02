#!/usr/bin/env python3
"""Download a model from HuggingFace with revision.

> python3 tools/download_model.py -m <HF-model-id> [-r <git-tag-or-hash>]

"""

import argparse
import logging
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model", help="HuggingFace model ID")
    parser.add_argument(
        "-r", dest="revision", default="main", help="Git hash, tag, or branch (default='main')"
    )
    args, _extra_args = parser.parse_known_args()

    if args.model:
        logging.info("Downloading model '%s' with revision '%s' ...", args.model, args.revision)
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            ignore_patterns=[
                "onnx/*",
                "openvino/*",
                "*.msgpack",
                "*.h5",
                "*.ot",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            ],
        )
    else:
        logging.error("Need to provide a HuggingFace model ID.")
        exit(1)


if __name__ == "__main__":
    main()
