# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SenDNN Inference is a vLLM plugin that enables IBM Spyre Accelerator integration. It extends vLLM's scheduler, model worker, model runner, and modeling code to support Spyre hardware. The plugin registers via the `vllm.platform_plugins` entry point (`sendnn_inference:register` → `SpyrePlatform`).

## Development Setup

```bash
# Install uv and create venv
pip install uv
uv venv --python 3.12 --seed .venv --system-site-packages
source .venv/bin/activate

# Install from source with dev dependencies (handles torch CPU via uv sources/indexes)
uv sync --frozen --active --inexact
```

## Common Commands

```bash
# Format and lint (runs ruff, typos, ty type checker via `uvx prek`)
bash format.sh

# Format only specific files
bash format.sh --files path/to/file.py

# Run all tests
pytest

# Run tests with specific markers
pytest -m "cpu"             # CPU-only tests (most useful for local dev)
pytest -m "e2e"             # End-to-end tests
pytest -m "spyre"           # Spyre hardware tests (requires hardware)
pytest -m "chunked_prefill" # Chunked prefill tests

# Run single test file or function
pytest tests/e2e/test_spyre_basic.py
pytest tests/e2e/test_spyre_basic.py::test_specific_case
```

## Architecture

The plugin registers as a vLLM platform via entry point and overrides three components:

1. **Platform** (`sendnn_inference/platform.py`) - `SpyrePlatform(Platform)` — plugin entry point, config validation, request validation hooks, warmup shape management
2. **Scheduler** (`sendnn_inference/v1/core/scheduler.py`) - `SpyreScheduler(Scheduler)` subclasses handle batching constraints (static shapes for pooling, chunked prefill for decoders)
3. **Worker** (`sendnn_inference/v1/worker/`) - `SpyreWorker`, `SpyreModelRunner`, `SpyreInputBatch` — model execution on Spyre cards

### Key Directories

- `sendnn_inference/v1/` - vLLM V1 backend implementation
- `sendnn_inference/multimodal/mm_mappings/` - Multimodal model processors (LlavaNext, Mistral3)
- `sendnn_inference/config/` - Model configuration registry and YAML-based matching
- `sendnn_inference/model_executor/model_loader/` - Spyre-specific model loading
- `tests/` - Test suites organized by category (e2e, config, multimodal, v1)

## Configuration

### Environment Variables

All configuration uses `SENDNN_INFERENCE_*` prefix. Key variables:

- `SENDNN_INFERENCE_DYNAMO_BACKEND` - "sendnn" (hardware), "eager" (CPU debug)
- `SENDNN_INFERENCE_WARMUP_BATCH_SIZES` - Static batch sizes for pooling models
- `SENDNN_INFERENCE_WARMUP_PROMPT_LENS` - Prompt lengths for pooling warmup
- `SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS` - Enforce precompiled model cache
- `SENDNN_INFERENCE_CPU_MM_DTYPE` - Dtype for multimodal vision encoder on CPU

For CPU-only testing, source `_local_envs_for_test.sh` (sets `MASTER_ADDR`, `MASTER_PORT`, eager backend, `HF_HUB_OFFLINE=1`, disables multiprocessing).

### Model Configuration

Models are registered in `sendnn_inference/config/model_configs.yaml` with architecture patterns and device-specific configs (env vars, block overrides). New models require:

1. FMS implementation passing AFTU tests with paged attention + chunked prefill
2. Model loader entry in `sendnn_inference/model_executor/model_loader/spyre.py`
3. Configuration in `model_configs.yaml`

## Testing Notes

- `conftest.py` auto-parametrizes tests with `model` and `backend` fixtures — you don't need to add these params to test functions manually
- Tests are sorted for LLM caching: the test framework caches vLLM instances (`LLM`, server, engine) across tests with the same config to avoid redundant model loads
- Tests marked `fork_required` or using direct engine access run in forked subprocesses
- Tests use `HF_HUB_OFFLINE=1` with cached models in `tests/hf_cache.json`
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` for local debugging
- Spyre hardware tests require webhook-triggered validation suite (`bot:test` PR comment)

## Linting and Type Checking

Pre-commit hooks (run via `bash format.sh` / `uvx prek`):
- **ruff** — linting + formatting (line length 100)
- **typos** — spell checking
- **ty** — type checking (`sendnn_inference/` only, version pinned in `.pre-commit-config.yaml`)
- **actionlint** — GitHub Actions validation

## vLLM Compatibility

The codebase maintains backwards compatibility using `compat_utils.py`:
- `has_argument(func, "arg")` — conditionally pass arguments based on vLLM version
- Add tests that fail when backwards compatibility is no longer needed (so compat shims get cleaned up)
