---
name: update-vllm
description: Upgrade the sendnn-inference plugin to a newly released vLLM version. Bumps the pin in pyproject.toml, regenerates uv.lock, runs CPU + compat tests, adds compat shims if upstream APIs changed, and extends the CI backwards-compat matrix. Use when the user says "upgrade vllm to X.Y.Z", "bump vllm", "support the new vllm release", or similar.
argument-hint: <target-vllm-version e.g. 0.22.0>
---

Upgrade this plugin to vLLM `$ARGUMENTS`.

If `$ARGUMENTS` is empty, ask the user which target version before doing anything else.

**Follow the complete procedure in `docs/contributing/vllm-update-procedure.md`.**

When you see `{VERSION}` in that document, replace it with `$ARGUMENTS` (the version provided by the user or requested from them).
