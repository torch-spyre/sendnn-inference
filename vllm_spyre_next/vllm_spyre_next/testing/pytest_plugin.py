"""
pytest11 plugin for vllm-spyre-next.

This plugin integrates upstream vLLM tests with vLLM-spyre-next
and filtering via a declarative YAML config (upstream_tests.yaml).

Hook Execution Order
---------------------
1. pytest_configure (tryfirst)
    - Loads Spyre plugins (custom ops, platform)
    - Detects local vLLM repo OR clones to ~/.cache/vllm-upstream-tests/
    - Injects test paths into pytest collection

2. pytest_generate_tests (tryfirst)
    - Overrides test parameters from YAML (e.g., num_tokens: [1, 16])
    - Must run during collection before parametrization finalizes

3. pytest_collection_modifyitems
    - Applies skip/xfail markers based on YAML allow_list/block_list
    - Applies tag markers for filtering (e.g., pytest -m rmsnorm)

4. pytest_fixture_setup (tryfirst)
    - Overrides default_vllm_config fixture with Spyre-specific config

Environment Variables
---------------------
SKIP_UPSTREAM_TESTS     Set to 1/true/yes to skip upstream test cloning
UPSTREAM_TESTS_PATHS    Comma-separated paths (default: auto from YAML)
VLLM_COMMIT             Override vLLM commit (default: from pyproject.toml)
VLLM_REPO_URL           Override vLLM repo URL
XDG_CACHE_HOME          Base cache directory (default: ~/.cache)
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

from vllm_spyre_next.testing.models import (
    AllowEntry,
    BlockEntry,
    FileConfig,
    ParamAllow,
    ParamOverride,
    ParamSkip,
    UpstreamTestConfig,
)

_YAML_FILENAME = "upstream_tests.yaml"
_YAML_PATH = Path(__file__).parent / _YAML_FILENAME

# Global terminal reporter for pytest-aware logging
_terminal_reporter = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str):
    """Log message to pytest terminal reporter if available.
    This allows logs to be printed from this file even when pytest is capturing output.
    """
    if _terminal_reporter:
        _terminal_reporter.write_line(msg)
    else:
        # Fallback to stderr when terminal reporter not available
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# YAML Config Loading
# ---------------------------------------------------------------------------


def _load_upstream_config() -> UpstreamTestConfig:
    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f)
    if not raw or "tests" not in raw or "files" not in raw["tests"]:
        raise RuntimeError(
            f'Invalid YAML in {_YAML_PATH}: missing "tests" or "tests.files" sections'
        )
    return _parse_config(raw["tests"])


def _parse_config(raw_tests: dict) -> UpstreamTestConfig:
    files: list[FileConfig] = []
    for file_entry in raw_tests.get("files", []):
        allow_list: list[AllowEntry] = []
        for allow in file_entry.get("allow_list", []):
            params_section = allow.get("params", {})
            param_skips = [
                ParamSkip(param_name=k, values=frozenset(v))
                for k, v in params_section.get("skip", {}).items()
            ]
            param_allows = [
                ParamAllow(param_name=k, values=frozenset(v))
                for k, v in params_section.get("allow", {}).items()
            ]
            param_overrides = [
                ParamOverride(param_name=k, values=tuple(v))
                for k, v in params_section.get("override", {}).items()
            ]
            allow_list.append(
                AllowEntry(
                    test=allow["test"],
                    mode=allow.get("mode", "mandatory_pass"),
                    tags=tuple(allow.get("tags", [])),
                    param_skips=tuple(param_skips),
                    param_allows=tuple(param_allows),
                    param_overrides=tuple(param_overrides),
                )
            )
        block_list = [BlockEntry(test=b["test"]) for b in file_entry.get("block_list", [])]
        files.append(
            FileConfig(
                rel_path=file_entry["rel_path"],
                allow_list=tuple(allow_list),
                block_list=tuple(block_list),
            )
        )
    return UpstreamTestConfig(files=tuple(files))


_UPSTREAM_CONFIG: UpstreamTestConfig = _load_upstream_config()


def _get_paths_from_yaml() -> str:
    """Extract test paths from upstream_tests.yaml rel_path entries.

    Uses exact rel_path (file or folder), e.g.:
      "tests/kernels/core/test_layernorm.py" -> "kernels/core/test_layernorm.py"
      "tests/models/" -> "models/"
    """
    paths = []
    for fc in _UPSTREAM_CONFIG.files:
        p = Path(fc.rel_path)
        # Strip "tests/" prefix if present
        if p.parts and p.parts[0] == "tests":
            paths.append(str(Path(*p.parts[1:])))
        else:
            paths.append(str(p))
    return ",".join(paths)


# ---------------------------------------------------------------------------
# vLLM Repository Cloning
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    """
    Cache directory for cloned tests (persists across runs)
    """
    # Respect XDG if present, fallback to ~/.cache
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "vllm-upstream-tests"


def _extract_vllm_commit_from_pyproject() -> str:
    """
    Extract the vLLM git reference from pyproject.toml [tool.uv.sources] section.
    Raises FileNotFoundError if pyproject.toml is missing, or KeyError
    if the expected source entry is not found.
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(
            f"pyproject.toml not found in {Path(__file__).parent.parent.parent}"
        )

    content = pyproject_path.read_text()
    # Look for vllm source with git and rev
    # Pattern: vllm = { git = "...", rev = "commit_sha_or_semver_tag" }
    match = re.search(
        r'vllm\s*=\s*\{\s*git\s*=\s*"[^"]+"\s*,\s*rev\s*=\s*"([0-9a-f]{7,40}|v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?)"\s*\}',
        content,
    )
    if match:
        return match.group(1)

    raise KeyError("Ensure vllm is specified with 'rev' in pyproject.toml [tool.uv.sources]")


def _resolve_vllm_commit() -> str:
    """
    Resolve the vLLM git reference to use for cloning upstream tests.
    Priority: VLLM_COMMIT env var > pyproject.toml > error
    """
    # Allow env var override for testing/CI
    env_commit = os.environ.get("VLLM_COMMIT", "").strip()
    if env_commit:
        if not re.match(r"^(?:[0-9a-f]{7,40}|v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?)$", env_commit):
            raise ValueError(f"Invalid VLLM_COMMIT format: {env_commit}")
        return env_commit

    # Extract from pyproject.toml
    return _extract_vllm_commit_from_pyproject()


def _run(cmd: list[str], cwd: Path | None = None, max_retries: int = 3) -> None:
    """Run command with optional retries for network operations."""
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                raise


def _ensure_repo_at_commit(repo_dir: Path, url: str, commit: str, sparse_paths: list[str]) -> Path:
    """
    Ensure repo cloned at 'repo_dir/commit' with sparse checkout of 'sparse_paths'.
    Returns the path to the working tree at that commit.
    """
    # We create a separate worktree per commit to allow co-existence of different commits
    base_dir = repo_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    git_dir = base_dir / "repo.git"

    if not git_dir.exists():
        _run(["git", "init", "--bare", str(git_dir)])

    # Prepare a worktree dir per commit
    wt_dir = base_dir / f"worktree-{commit[:12]}"
    if wt_dir.exists():
        _log(f"[vllm-upstream] Using cached worktree at {wt_dir}")
        return wt_dir

    # Create temp dir to set up the sparse worktree then move into place atomically
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as td:
        td_path = Path(td)

        # Ensure origin remote exists and points to the correct URL
        result = subprocess.run(
            ["git", "--git-dir", str(git_dir), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Origin doesn't exist - add it
            _run(["git", "--git-dir", str(git_dir), "remote", "add", "origin", url])
        elif result.stdout.strip() != url:
            # Origin exists but points to different URL - update it
            _log(f"[vllm-upstream] Updating origin URL: {result.stdout.strip()} -> {url}")
            _run(["git", "--git-dir", str(git_dir), "remote", "set-url", "origin", url])

        # Determine if commit is a tag (starts with 'v' and matches semver pattern) or a SHA
        is_tag = re.match(r"^v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$", commit)

        if is_tag:
            _log(f"[vllm-upstream] Fetching tag {commit} from {url}")
            # For tags, fetch the tag reference
            _run(
                [
                    "git",
                    "--git-dir",
                    str(git_dir),
                    "fetch",
                    "--depth=1",
                    "origin",
                    f"refs/tags/{commit}:refs/tags/{commit}",
                ]
            )
        else:
            _log(f"[vllm-upstream] Fetching commit {commit[:12]} from {url}")
            # For commit SHAs, fetch the commit directly
            _run(["git", "--git-dir", str(git_dir), "fetch", "--depth=1", "origin", commit])

        # Create a new worktree at temp
        # For tags, use the full tag reference; for commits, use the commit SHA directly
        worktree_ref = f"refs/tags/{commit}" if is_tag else commit
        _run(
            [
                "git",
                "--git-dir",
                str(git_dir),
                "worktree",
                "add",
                "--detach",
                str(td_path),
                worktree_ref,
            ]
        )

        # Enable sparse checkout at the worktree
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=td_path)
        _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=td_path)

        # Ensure we're exactly at the commit (detached HEAD)
        _run(["git", "checkout", "--detach", commit], cwd=td_path)

        # Atomically move into place
        td_path.rename(wt_dir)

    return wt_dir


def _prepare_upstream_tests_dir() -> Path:
    """Clone vLLM to cache and return path to tests directory."""
    commit = _resolve_vllm_commit()
    cache_root = _cache_root()
    wt_dir = _ensure_repo_at_commit(
        repo_dir=cache_root,
        url=os.environ.get("VLLM_REPO_URL", "https://github.com/vllm-project/vllm"),
        commit=commit,
        sparse_paths=["tests"],
    )
    tests_dir = wt_dir / "tests"
    if not tests_dir.is_dir():
        raise RuntimeError(f"Upstream tests directory not found at {tests_dir}")
    return tests_dir


# ---------------------------------------------------------------------------
# Pytest Hooks
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Register Spyre plugins and detect/clone vLLM repo."""
    global _terminal_reporter
    _terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")

    # Set env vars BEFORE any vllm imports
    os.environ["VLLM_PLUGINS"] = "spyre_next,spyre_next_ops"
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"

    # Load plugins early to register custom ops before test modules import RMSNorm
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    # Detect local vLLM repo or clone it
    rootdir = Path(config.rootdir)
    tests_dir = rootdir / "tests"
    vllm_pkg = rootdir / "vllm"

    if tests_dir.is_dir() and vllm_pkg.is_dir():
        # Running from vLLM repo itself
        config._upstream_tests_base = tests_dir
        _log("[vllm-upstream] Using local vLLM tests")
    else:
        # Not in vLLM repo - check if we should clone
        skip_upstream = os.environ.get("SKIP_UPSTREAM_TESTS", "").lower() in ("1", "true", "yes")
        if skip_upstream:
            _log("[vllm-upstream] SKIP_UPSTREAM_TESTS is set, skipping upstream test collection")
            config._upstream_tests_base = None
            return

        try:
            # Clone vLLM to cache
            upstream_tests_base = _prepare_upstream_tests_dir()
            config._upstream_tests_base = upstream_tests_base

            # Determine which test paths to inject
            paths_env = _get_paths_from_yaml()
            _log(f"[vllm-upstream] Auto-derived test paths from YAML: {paths_env}")

            if not paths_env:
                _log("[vllm-upstream] No test paths configured, skipping upstream tests")
                config._upstream_tests_base = None
                return

            # Inject test paths into pytest collection
            for rel_path in paths_env.split(","):
                rel_path = rel_path.strip()
                if not rel_path:
                    continue
                test_dir = upstream_tests_base / rel_path
                if test_dir.exists():
                    _log(f"[vllm-upstream] Including tests from: {rel_path}")
                    config.args.append(str(test_dir))
                else:
                    _log(f"[vllm-upstream] Warning: Path not found: {test_dir}")

        except Exception as e:
            raise SystemExit(f"[vllm-upstream] Failed to prepare upstream tests: {e}") from e


# ---------------------------------------------------------------------------
# YAML Filtering Helpers
# ---------------------------------------------------------------------------


def _find_file_config(test_path: Path, file_configs: dict[Path, FileConfig]) -> FileConfig | None:
    if test_path in file_configs:
        return file_configs[test_path]
    for config_path, fc in file_configs.items():
        if test_path.is_relative_to(config_path):
            return fc
    return None


def _matches_block_list(test_name: str, block_list: tuple[BlockEntry, ...]) -> bool:
    return any(fnmatch.fnmatch(test_name, e.test) for e in block_list)


def _find_allow_entry(test_name: str, allow_list: tuple[AllowEntry, ...]) -> AllowEntry | None:
    for entry in allow_list:
        if fnmatch.fnmatch(test_name, entry.test):
            return entry
    return None


def _should_skip_params(item: pytest.Item, allow_entry: AllowEntry) -> bool:
    """Check if test should be skipped based on param_skips or param_allows.

    If param_allows is specified for a parameter, only those values are allowed.
    Otherwise, param_skips is used to exclude specific values.
    """
    callspec = getattr(item, "callspec", None)
    if not callspec:
        return False

    # Check param_allows first (whitelist takes precedence)
    for pa in allow_entry.param_allows:
        if pa.param_name in callspec.params and callspec.params[pa.param_name] not in pa.values:
            # If allowlist exists for this param, skip if value is NOT in allowlist
            return True

    # Check param_skips (blacklist)
    for ps in allow_entry.param_skips:
        if ps.param_name in callspec.params and callspec.params[ps.param_name] in ps.values:
            return True

    return False


# ---------------------------------------------------------------------------
# Collection Modification
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply YAML-based filtering to upstream tests and reorder tests."""
    upstream_tests_base = getattr(config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        # Still reorder tests even if not running upstream tests
        _reorder_tests_by_name(items)
        return

    upstream_tests_base = Path(upstream_tests_base).resolve()
    upstream_repo_root = upstream_tests_base.parent
    file_configs = {
        (upstream_repo_root / fc.rel_path).resolve(): fc for fc in _UPSTREAM_CONFIG.files
    }

    upstream_marker = pytest.mark.upstream

    for item in items:
        test_path = Path(item.fspath).resolve()
        if not test_path.is_relative_to(upstream_tests_base):
            continue

        item.add_marker(upstream_marker)

        fc = _find_file_config(test_path, file_configs)
        if fc is None:
            item.add_marker(pytest.mark.skip(reason=f"not in {_YAML_FILENAME}"))
            continue

        test_name = item.originalname or item.name
        if _matches_block_list(test_name, fc.block_list):
            item.add_marker(pytest.mark.skip(reason=f"blocked by {_YAML_FILENAME}"))
            continue

        allow_entry = _find_allow_entry(test_name, fc.allow_list)

        if allow_entry:
            for tag in allow_entry.tags:
                item.add_marker(getattr(pytest.mark, tag))

        if allow_entry is None:
            item.add_marker(pytest.mark.skip(reason="not in allow_list"))
            continue

        if _should_skip_params(item, allow_entry):
            item.add_marker(pytest.mark.skip(reason="param skipped"))
            continue

        if allow_entry.mode == "xfail":
            item.add_marker(pytest.mark.xfail(strict=False))
        elif allow_entry.mode == "xfail_strict":
            item.add_marker(pytest.mark.xfail(strict=True))

    # Reorder tests so that tests with "uses_subprocess" marker run first
    _reorder_tests_by_name(items)


def _reorder_tests_by_name(items: list[pytest.Item]) -> None:
    """Reorder tests so that tests with 'uses_subprocess' marker run first.

    This modifies the items list in-place using a stable sort, so tests marked
    with 'uses_subprocess' will run first while preserving the relative order
    within each group.
    """
    stable_map = {item: idx for idx, item in enumerate(items)}

    def sort_key(item: pytest.Item) -> tuple[int, int]:
        # Check if the test has the 'uses_subprocess' marker
        has_subprocess_marker = any(
            marker.name == "uses_subprocess" for marker in item.iter_markers()
        )

        # Priority 0: tests with uses_subprocess marker run first
        # Priority 1: all other tests
        priority = 0 if has_subprocess_marker else 1

        return (priority, stable_map[item])

    items.sort(key=sort_key)


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Apply parameter overrides from YAML config."""
    upstream_tests_base = getattr(metafunc.config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    upstream_repo_root = Path(upstream_tests_base).resolve().parent
    test_path = Path(metafunc.definition.fspath).resolve()
    file_configs = {
        (upstream_repo_root / fc.rel_path).resolve(): fc for fc in _UPSTREAM_CONFIG.files
    }

    fc = _find_file_config(test_path, file_configs)
    if not fc:
        return

    test_name = metafunc.definition.originalname or metafunc.definition.name
    allow_entry = _find_allow_entry(test_name, fc.allow_list)
    if not allow_entry or not allow_entry.param_overrides:
        return

    for po in allow_entry.param_overrides:
        if po.param_name not in metafunc.fixturenames:
            continue
        for i, marker in enumerate(metafunc.definition.own_markers):
            if marker.name == "parametrize" and marker.args[0] == po.param_name:
                metafunc.definition.own_markers[i] = pytest.mark.parametrize(
                    po.param_name, list(po.values)
                ).mark
                break


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spyre_default_vllm_config(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.forward_context import set_forward_context

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)

    # Explicitly register custom ops
    from vllm_spyre_next.custom_ops import register_all

    register_all()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        # Set forward context so custom ops can access the vllm config
        yield


@pytest.fixture()
def default_vllm_config(monkeypatch):
    yield from _spyre_default_vllm_config(monkeypatch)


@pytest.fixture()
def should_do_global_cleanup_after_test():
    """Skip global cleanup for Spyre - torch.accelerator.empty_cache() doesn't work yet."""
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    """Override fixtures when running upstream vLLM tests."""
    upstream_tests_base = getattr(request.config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    if fixturedef.argname == "default_vllm_config":
        fixturedef.func = _spyre_default_vllm_config
        fixturedef.argnames = ("monkeypatch",)
    elif fixturedef.argname == "should_do_global_cleanup_after_test":
        fixturedef.func = lambda: False
        fixturedef.argnames = ()
