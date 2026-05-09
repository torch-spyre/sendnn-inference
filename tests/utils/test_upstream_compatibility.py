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
