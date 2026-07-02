#!/usr/bin/env python3
"""Build a nano-scale random-init granite-vision variant for CI tests.

Produces a `LlavaNextForConditionalGeneration` snapshot with:

  - `model_type: llava_next` (passes ``LlavaNextMMUtils._validate_configs``)
  - `text_config.model_type: granite` (same)
  - `vision_config.model_type: siglip_vision_model` (only backbone FMS supports)
  - Tiny dims (text hidden 64 × 2 layers; vision hidden 64 × 4 layers)
  - Random init — outputs are meaningless, only wiring matters
  - Tokenizer + processor copied from a source snapshot so
    ``image_token_index`` and chat template stay valid

Usage:

    # Build locally
    python3 tools/build_nano_gv.py --out /tmp/nano-gv

    # Build and push to HF Hub (requires `huggingface-cli login` first)
    python3 tools/build_nano_gv.py --out /tmp/nano-gv --publish joerunde/nano-gv

The output directory can then be passed to ``LLM(model=...)`` or
``vllm serve`` as a local model path (or via the HF id after --publish).

Model on disk is ~16 MB — fits comfortably alongside the other CI cache
entries.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, LlavaNextForConditionalGeneration


# Files carried over from the source snapshot (tokenizer + preprocessor state).
# We keep the full tokenizer so `image_token_index=49155` and the chat template
# stay valid — shrinking the tokenizer would break the whole processor pipeline
# and gives us at most a few MB back.
_TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def _make_nano_config(source_config):
    """Return a shrunken copy of the source LlavaNextConfig.

    Preserved fields the plugin / FMS looks at:
      - top-level: model_type, image_token_index, image_grid_pinpoints (kept
        as a single entry matching image_size), vision_feature_layer,
        vision_feature_select_strategy, tie_word_embeddings,
        use_image_newline_parameter
      - text_config: model_type=granite plus all the granite-specific
        multipliers (attention_multiplier, embedding_multiplier,
        logits_scaling, residual_multiplier); shrunk dims
      - vision_config: model_type=siglip_vision_model plus shrunk dims;
        image_size aligned to patch_size

    The `vision_feature_layer=[-4,-3,-2,-1]` pattern is preserved so the
    multi-modal projector input dim (hidden * 4) matches what the FMS
    LlavaNext implementation expects. That forces `num_hidden_layers >= 4`
    on the vision tower.
    """
    cfg = source_config.to_dict()

    # ── vision tower ──────────────────────────────────────────────────────
    # 4 layers so `vision_feature_layer=[-4,-3,-2,-1]` remains valid.
    # image_size=112 with patch_size=14 → 8×8=64 patches. Tiny but still a
    # real image tensor.
    cfg["vision_config"] = {
        **cfg["vision_config"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "image_size": 112,
        "patch_size": 14,
    }
    # image_grid_pinpoints must include at least the base image_size so the
    # processor can tile a plain 112×112 input.
    cfg["image_grid_pinpoints"] = [[112, 112], [112, 224], [224, 112]]
    # `vision_feature_layer` in the source is [-24, -20, -12, -1] — indices
    # into a 27-layer vision tower. With num_hidden_layers=4 those are out
    # of range and produce IndexError in fms/models/llava_next.py:348.
    # Keep four features (so the multi_modal_projector input dim stays at
    # `vision_hidden * 4 = 256` and matches the checkpoint), but reference
    # every layer in the 4-layer stack.
    cfg["vision_feature_layer"] = [-4, -3, -2, -1]

    # ── text backbone ─────────────────────────────────────────────────────
    # `num_attention_heads=2` is the minimum that satisfies vLLM's TP=2
    # divisibility check. hidden_size=64 with head_dim=32 keeps Q/K/V/O
    # projections symmetric (64→64 in and out) so FMS's weight-expansion
    # logic doesn't kick in.
    #
    # `head_dim` is set explicitly so that
    # `LlavaNextMMUtils.get_mm_specific_load_overrides` sees an explicit
    # value and skips its `head_dim=128` fallback (which would otherwise
    # clobber the entire text_config with a partial dict). See the plugin
    # file for the details.
    cfg["text_config"] = {
        **cfg["text_config"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        # vocab_size unchanged — must match the tokenizer we copy over.
    }

    # Preserve `tie_word_embeddings=True` on the text config too — the FMS
    # loader reads `config.tie_word_embeddings` and the safetensors index
    # for the tied case only stores one copy.
    cfg["text_config"]["tie_word_embeddings"] = True

    from transformers import LlavaNextConfig

    return LlavaNextConfig(**cfg)


def build(source_model: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nano-gv] loading source config from {source_model!r}")
    source_config = AutoConfig.from_pretrained(source_model)

    print("[nano-gv] shrinking config")
    nano_config = _make_nano_config(source_config)

    print("[nano-gv] instantiating LlavaNextForConditionalGeneration with random weights")
    # dtype=bfloat16 matches the source model's `torch_dtype` — keeps the
    # plugin's fp16 cast path deterministic (bf16 params → fp16 for Spyre).
    torch.manual_seed(0)
    model = LlavaNextForConditionalGeneration(nano_config).to(dtype=torch.bfloat16)

    print(f"[nano-gv] saving model to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)

    # Copy tokenizer + processor state from source, then re-save so all
    # image-preprocessor dimensions match our nano vision_config. The
    # source processor's crop_size / size / image_grid_pinpoints all
    # reference the source image_size (384) — if we don't update them,
    # vLLM's dummy-input generator will build 384×384 pixel tensors that
    # produce 27²=729 patches, and the nano vision tower (expecting 8²=64
    # patches from a 112×112 image) will fail at forward time.
    print("[nano-gv] copying tokenizer + processor from source")
    processor = AutoProcessor.from_pretrained(source_model)

    img_size = nano_config.vision_config.image_size
    if hasattr(processor, "image_processor"):
        proc = processor.image_processor
        if hasattr(proc, "crop_size"):
            proc.crop_size = {"height": img_size, "width": img_size}
        if hasattr(proc, "size"):
            proc.size = {"height": img_size, "width": img_size}
        # image_grid_pinpoints must be a subset (or match) of the top-level
        # config's image_grid_pinpoints. Use the same tiny list.
        if hasattr(proc, "image_grid_pinpoints"):
            proc.image_grid_pinpoints = list(nano_config.image_grid_pinpoints)

    processor.save_pretrained(out_dir)

    # AutoProcessor.save_pretrained handles most of _TOKENIZER_FILES, but not
    # merges.txt / vocab.json for BPE tokenizers — copy manually if missing.
    src_snapshot = (
        Path(processor.tokenizer.vocab_file).parent
        if getattr(processor.tokenizer, "vocab_file", None)
        else None
    )
    if src_snapshot is not None:
        for fname in ("merges.txt", "vocab.json"):
            src = src_snapshot / fname
            dst = out_dir / fname
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"[nano-gv]   copied {fname}")

    _write_readme(out_dir, source_model)

    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    print(f"[nano-gv] done — {total / 1024**2:.1f} MB in {out_dir}")


def _write_readme(out_dir: Path, source_model: str) -> None:
    """Drop a short model card next to the weights.

    HF Hub's model-card linter warns on repos without a README, and it's
    the right place to record that this is a CI fixture (random weights)
    rather than something anyone should use for inference.
    """
    readme = out_dir / "README.md"
    readme.write_text(
        f"""---
license: apache-2.0
tags:
  - test
  - fixture
---

# nano-gv

Random-init ~16 MB CI fixture for the [sendnn-inference](https://github.com/torch-spyre/sendnn-inference)
plugin's async multimodal-encoder tests. Generated by `tools/build_nano_gv.py`
from [`{source_model}`](https://huggingface.co/{source_model})'s config.

**Not a functional model — outputs are meaningless.** Do not use for
inference. The architecture identifiers (`LlavaNextForConditionalGeneration`
+ granite text + siglip vision) match the source, but every dim is
shrunk to the minimum that still exercises the plugin's MM code paths:

| component | source           | nano                |
|-----------|------------------|---------------------|
| text hidden      | 2048       | 64                  |
| text layers      | 40         | 2                   |
| text heads       | 32         | 2                   |
| text head_dim    | 64         | 32                  |
| vision hidden    | 1152       | 64                  |
| vision layers    | 27         | 4                   |
| image size       | 384        | 112                 |

The tokenizer + processor are copied from the source verbatim (with
image-size fields rescaled to 112) so `image_token_index` and the chat
template stay valid.
"""
    )
    print(f"[nano-gv] wrote README.md")


def _publish(local_dir: Path, repo_id: str, private: bool, message: str) -> None:
    """Upload the local snapshot to HF Hub at *repo_id*.

    Creates the repo if it doesn't exist. Requires an HF write token —
    run `huggingface-cli login` first, or set HF_TOKEN in the environment.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"[nano-gv] publishing to https://huggingface.co/{repo_id}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        repo_type="model",
        commit_message=message,
    )
    print(f"[nano-gv] published: https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="ibm-granite/granite-vision-3.2-2b",
        help="HF model id to derive the config + tokenizer from",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the nano model snapshot",
    )
    parser.add_argument(
        "--publish",
        metavar="REPO_ID",
        default=None,
        help=(
            "After building, upload the snapshot to HF Hub at REPO_ID "
            "(e.g. 'joerunde/nano-gv'). Requires `huggingface-cli login` "
            "or HF_TOKEN in the environment."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If publishing, create the repo as private.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update nano-gv snapshot",
        help="Commit message for the HF Hub upload.",
    )
    args = parser.parse_args()

    build(args.source, args.out)

    if args.publish:
        _publish(args.out, args.publish, args.private, args.commit_message)


if __name__ == "__main__":
    main()
