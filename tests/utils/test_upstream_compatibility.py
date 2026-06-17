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
