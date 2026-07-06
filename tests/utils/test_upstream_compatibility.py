"""
Tests checking for vLLM upstream compatibility requirements.

As we remove support for old vLLM versions, we want to keep track of the
compatibility code that can be cleaned up.
"""

import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_compilation_times_compat():
    """
    When this test starts failing because CompilationTimes exists in the lowest supported vllm
    version, the try/except import and conditional usage of CompilationTimes in
    spyre_worker.py can be simplified to an unconditional import.
    """
    import vllm.v1.worker.worker_base as worker_base

    if VLLM_VERSION == "vLLM:lowest":
        assert not hasattr(worker_base, "CompilationTimes"), (
            "Backwards compatibility shim for CompilationTimes in spyre_worker.py can be removed"
        )


def test_kv_cache_manager_scheduler_block_size_compat():
    """
    When this test starts failing because KVCacheManager.__init__ requires `scheduler_block_size`
    in the lowest supported vllm version, the conditional has_argument check in
    tests/llm_cache.py `_reset_scheduler` can be replaced with an unconditional kwarg.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from sendnn_inference.compat_utils import has_argument

    if VLLM_VERSION == "vLLM:lowest":
        assert not has_argument(KVCacheManager.__init__, "scheduler_block_size"), (
            "Backwards compatibility shim for scheduler_block_size in tests/llm_cache.py "
            "can be removed"
        )


def test_scheduler_schedule_throttle_prefills_compat():
    """
    In vLLM >= 0.24.0, Scheduler.schedule() gained a `throttle_prefills: bool`
    parameter. SpyreScheduler.schedule() and ChunkedPrefillSpyreScheduler.schedule()
    must accept and forward it.

    When this test fails (lowest supported vllm already has throttle_prefills),
    the default value in both overrides can be dropped and the signature
    simplified to match upstream exactly.
    """
    from vllm.v1.core.sched.interface import SchedulerInterface

    from sendnn_inference.compat_utils import has_argument

    if VLLM_VERSION == "vLLM:lowest":
        assert not has_argument(SchedulerInterface.schedule, "throttle_prefills"), (
            "Backwards compatibility shim for throttle_prefills in "
            "sendnn_inference/v1/core/scheduler.py can be removed — "
            "lowest vLLM version already has the parameter."
        )


def test_tokenizer_registry_circular_import_compat():
    """
    In vLLM >= 0.24.0, vllm.utils.torch_utils imports is_pin_memory_available at module-level,
    creating a circular import when platform.py tries to patch vllm.tokenizers.registry at
    module-level during platform resolution. The shim in platform.py guards the module-level
    call with a sys.modules check and defers to check_and_update_config instead.

    When this test fails (lowest supported vllm no longer has the module-level
    is_pin_memory_available import in torch_utils.py), the sys.modules guard in platform.py
    can be removed and the unconditional module-level call restored.
    """
    import importlib
    import sys

    if VLLM_VERSION == "vLLM:lowest":
        # On the lowest supported version, torch_utils should NOT import
        # is_pin_memory_available at module-level (i.e. the guard is not needed yet)
        if "vllm.utils.torch_utils" in sys.modules:
            torch_utils = sys.modules["vllm.utils.torch_utils"]
        else:
            torch_utils = importlib.import_module("vllm.utils.torch_utils")
        import inspect

        source = inspect.getsource(torch_utils)
        assert "from vllm.utils.platform_utils import is_pin_memory_available" not in source, (
            "vLLM:lowest now imports is_pin_memory_available at module-level in torch_utils.py. "
            "The sys.modules guard in the platform.py module-level patch can be removed and "
            "SpyrePlatform._patch_tokenizer_registry_get_config() called unconditionally again."
        )
