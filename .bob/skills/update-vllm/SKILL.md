---
name: update-vllm
description: Upgrade the sendnn-inference plugin to a newly released vLLM version. Bumps the pin in pyproject.toml, regenerates uv.lock, runs CPU + compat tests, adds compat shims if upstream APIs changed, and extends the CI backwards-compat matrix. Use when the user says "upgrade vllm to X.Y.Z", "bump vllm", "support the new vllm release", or similar.
---

Upgrade this plugin to the vLLM version specified by the user.

**IMPORTANT**: If the user hasn't specified a target version, ask them which version before proceeding.

**Follow the complete procedure in `docs/contributing/vllm-update-procedure.md`.**

When you see `{VERSION}` in that document, replace it with the actual version number provided by the user (e.g., `0.22.0`).
