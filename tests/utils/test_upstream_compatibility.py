"""
Tests checking for vLLM upstream compatibility requirements.

As we remove support for old vLLM versions, we want to keep track of the
compatibility code that can be cleaned up.
"""

import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_tokenizer_registry_get_config_patch():
    """
    When this test starts failing because get_config exists in the lowest supported vllm version,
    the check to conditionally _not_ apply _patch_tokenizer_registry_get_config can be removed.
    """
    import vllm.tokenizers.registry as tokenizer_registry

    # Check if get_config exists in the tokenizer registry module
    # (it was added in vllm 0.19.1)
    if VLLM_VERSION == "vLLM:lowest":
        assert not hasattr(tokenizer_registry, "get_config"), (
            "Backwards compatibility code in _patch_tokenizer_registry_get_config can be removed"
        )
