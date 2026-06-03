---
name: update-vllm
description: Upgrade the sendnn-inference plugin to a newly released vLLM version. Bumps the pin in pyproject.toml, regenerates uv.lock, runs CPU + compat tests, adds compat shims if upstream APIs changed, and extends the CI backwards-compat matrix. Use when the user says "upgrade vllm to X.Y.Z", "bump vllm", "support the new vllm release", or similar.
---

Upgrade this plugin to the vLLM version specified by the user. The procedure below is the distilled checklist; it has produced clean upgrades for past patch releases (0.20.1, 0.20.2, 0.21.0) and surfaced the right compat shims for breaking releases (0.20.0, 0.10.1).

**IMPORTANT**: If the user hasn't specified a target version, ask them which target version before doing anything else. Once you have the version, use it consistently throughout all commands by replacing `<VERSION>` with the actual version number (e.g., `0.22.0`).

The companion doc at `docs/contributing/maintaining.md` is the source of truth for the high-level workflow. This skill is the operational checklist: exact commands, ordered tests, and the compat-shim playbook.

## Phase 0 — Preflight

1. Verify the upstream tag exists. If this returns 404, stop — the user has the version wrong:

   ```bash
   curl -fsS https://api.github.com/repos/vllm-project/vllm/git/refs/tags/v<VERSION>
   ```

   Replace `<VERSION>` with the actual version number provided by the user (e.g., `0.22.0`).

2. Detect hardware mode:

   ```bash
   echo "AIU_WORLD_SIZE=${AIU_WORLD_SIZE:-<unset>}"
   ```

   - **Unset** → CPU/eager mode. All test invocations must use `-m "cpu"` (and source `_local_envs_for_test.sh`). Spyre-marked tests are deferred to the post-PR `bot:test` validation suite.
   - **Set** → hardware available. After CPU tests pass, optionally also run `-m "spyre"` markers.

3. Branch from `main` if not already on a `vllm-<VERSION>` branch:

   ```bash
   git checkout -b vllm-<VERSION> main
   ```

   Replace `<VERSION>` with the actual version number provided by the user (e.g., `vllm-0.22.0`).

4. Read the upstream release notes and skim for breaking-change candidates:

   ```bash
   gh api repos/vllm-project/vllm/releases/tags/v<VERSION> --jq '.body' | head -200
   ```

   Replace `<VERSION>` with the actual version number provided by the user.

   High-impact areas to flag for this plugin:
   - **Worker / model runner V2** (`vllm.v1.worker.worker_base.WorkerBase`, `compile_or_warm_up_model` return type, `KVCacheConfig`, `KVCacheSpec`)
   - **Scheduler / engine core** (`SchedulerOutput`, `CachedRequestData`, `NewRequestData`, MRv2 changes, KV connectors)
   - **Platform interface** (`vllm.platforms.Platform` — new abstract methods, removed methods)
   - **Sampler / output dataclasses** (`ModelRunnerOutput`, `SamplerOutput`)
   - **Tokenizer registry** (this plugin patches it in `platform.py`)
   - **Multimodal processor APIs** (LlavaNext, Mistral3 → `sendnn_inference/multimodal/mm_mappings/`)
   - **Configuration** (`ModelConfig`, `SchedulerConfig`, `VllmConfig` — new/removed fields)
   - **Removed / deprecated APIs** — for each, grep this repo to see if it's still referenced.

## Phase 1 — Bump the pin

Edit `pyproject.toml`:

1. `[project.dependencies]`: bump the upper bound to `<X.Y.(Z+1)` (e.g., 0.22.0 → `<0.22.1`). Keep the existing lower bound unless explicitly raising it.
2. `[tool.uv.sources.vllm].rev`: set to `vX.Y.Z`.
3. **Audit `[tool.uv].override-dependencies`**:
   - Re-fetch upstream `requirements/common.txt` and check whether each override is still load-bearing:

     ```bash
     curl -fsS https://raw.githubusercontent.com/vllm-project/vllm/v<VERSION>/requirements/common.txt | grep -E "llguidance|torch|transformers"
     ```

     Replace `<VERSION>` with the actual version number.

   - If vLLM bumped torch, update the `torch==X.Y.Z` override to match. (vLLM 0.20.0 bumped 2.10→2.11, for example.)
   - If a forced floor (e.g., `llguidance>=1.7.3`) is now within vLLM's own range, the override may be relaxable — but only remove it if the original reason (CVE, arch fix) is genuinely resolved upstream.

Then regenerate the lockfile and reinstall:

```bash
source .venv/bin/activate    # if not already active
uv sync --active --inexact
bash format.sh
```

## Phase 2 — Run CPU + compat tests

Always source the env file first; without it `MASTER_ADDR`/`MASTER_PORT`/`SENDNN_INFERENCE_DYNAMO_BACKEND=eager` are unset and tests will fail or use the wrong backend:

```bash
source _local_envs_for_test.sh
```

Run the suites in this order (cheapest → most expensive). Stop and fix on the first failure:

| # | Command | What it covers |
| --- | --- | --- |
| 1 | `pytest -m "cpu and basic and not quantized" --timeout=300 -x` | Smoke — fastest signal |
| 2 | `pytest -m "compat or (cpu and basic and not quantized)" --timeout=300` | Existing compat shims still work |
| 3 | `pytest -m "cpu and decoder and not quantized and not multimodal" --timeout=300` | Chunked prefill + prefix caching |
| 4 | `pytest -m "cpu and embedding and not quantized" --timeout=300` | Embedding |
| 5 | `pytest -m "cpu and scoring" --timeout=300` | Scoring |
| 6 | `pytest -m "not e2e and not quantized and not spyre and not multimodal" --timeout=300` | Worker + utils |
| 7 | `HF_HUB_OFFLINE=0 pytest -m "cpu and multimodal" --timeout=600` | Multimodal (needs HF online) |

If everything passes here with no source changes, this is a clean upgrade — skip Phase 3.

## Phase 3 — Compat shims (only when tests fail)

Use `sendnn_inference/compat_utils.py` helpers (`has_argument`, `dataclass_fields`) plus the patterns below. **For every shim added, add a corresponding test in `tests/utils/test_upstream_compatibility.py`** that fails when `VLLM_VERSION == "vLLM:lowest"` no longer needs it. The existing `test_compilation_times_compat` is the template.

| Symptom | Fix |
| --- | --- |
| `ImportError` for moved/removed symbol | `try: from new import X; except ImportError: X = None` (or `from old import X` in the except branch). Branch on `X is None` at call sites. See `spyre_worker.py` `CompilationTimes` shim. |
| Method gained or lost a kwarg | `kwargs = {"k": v} if has_argument(func, "k") else {}; func(..., **kwargs)` |
| Dataclass added/removed a field | `if "field" in dataclass_fields(Cls): ...` |
| New abstract method on `Platform` / `WorkerBase` | Implement a stub on the Spyre subclass. See `manual_seed_all` added in PR #952. |
| Method signature changed return type | Wrap call site to coerce: `t = old_func(); return New(t) if New is not None else t` |
| Old `hasattr` guard now redundant | Delete the guard, simplify the call site, and **remove the matching compat test**. |

When in doubt, check past upgrade PRs for precedent: `gh pr list --search "vllm" --state merged --limit 30`. PR #952 (0.20.0) is the richest example.

## Phase 4 — Extend the CI matrix

In `.github/workflows/test.yml`, add a new entry under `include:` for the **previous** version (the old default). Mirror the format of the existing `vLLM:0.20.x` entries — only `name` and `repo` change. Keep `vLLM:lowest`, `vLLM:main`, and existing intermediates.

## Phase 5 — Verify and audit

1. Re-run `bash format.sh`. (ruff, typos, ty, actionlint should all pass.)
2. Audit `git diff --stat uv.lock`:
   - **Tiny** (~10 lines) → patch bump, only the vLLM rev moved. Expected.
   - **Huge** (~3000+ lines) → vLLM minor release reshuffled transitives. Expected.
   - **Medium** (50–500 lines) → look closely. Make sure unrelated packages didn't drift in surprising ways.
3. Verify only expected files changed:

   ```bash
   git status
   # Expected: pyproject.toml, uv.lock, .github/workflows/test.yml,
   #           and (if compat shims added) sendnn_inference/**, tests/utils/test_upstream_compatibility.py
   ```

## Phase 6 — Open the PR

Title: `chore: bumps vllm to v<VERSION>` (replace `<VERSION>` with actual version provided by the user)

Body skeleton:

```markdown
## Description
- Bumps vllm support to v<VERSION> (release notes: https://github.com/vllm-project/vllm/releases/tag/v<VERSION>)
- <List each compat shim added/removed and why, or "no source changes required">
- <List override-dependency changes, if any>

## Test Plan
- CPU + compat suites pass locally
- Hardware tests pending `bot:test` (run after PR opens)

## Checklist
- [x] I have read the contributing guidelines
- [x] My code follows the project's code style (run `bash format.sh`)
- [ ] I have added tests for my changes (if applicable)
- [ ] I have updated the documentation (if applicable)
- [x] My commits include a `Signed-off-by:` line (DCO compliance)
```

After the PR is open, request hardware validation by commenting:

```text
bot:test
MARKERS="spyre and not quantized"
```

For dependency-touching changes, comprehensive spyre-marked tests should be run before merge.

## Scope guardrails

- Do not bump the **lower bound** during a vLLM upgrade. Lower-bound bumps are a separate task (see `docs/contributing/maintaining.md` → "Removing support for a vLLM version") because they require deleting compat code and pruning the test matrix.
- Do not "improve" code outside the compat surface. A vLLM upgrade PR should be reviewable as one focused change.
- If a transitive dep moves in a surprising way (`uv sync` output shows e.g. transformers/torch shifting unexpectedly), surface it to the user before continuing — don't silently accept.
