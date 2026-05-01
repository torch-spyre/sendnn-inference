"""Shared fixtures for config tests."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sendnn_inference.config.model_registry import get_model_registry

# Shared path to test fixtures
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "model_configs"


@pytest.fixture(scope="session")
def registry():
    """Fixture providing a registry loaded with real model_configs.yaml."""
    return get_model_registry()


def _load_hf_config(fixture_path: Path) -> Mock:
    """Helper to load HF config from JSON and convert to Mock object."""
    with open(fixture_path) as f:
        config_dict = json.load(f)

    hf_config = Mock()
    for key, value in config_dict.items():
        setattr(hf_config, key, value)
    return hf_config


@pytest.fixture
def granite_3_3_hf_config():
    """Fixture providing real granite-3.3-8b-instruct HF config."""
    fixture_path = FIXTURES_PATH / "ibm-granite" / "granite-3.3-8b-instruct" / "config.json"
    return _load_hf_config(fixture_path)


@pytest.fixture
def granite_4_hf_dense_hybrid_config():
    """Fixture providing a version of a real granite-4-8b-dense HF config that's a spoofed version
    of a granitemoehybrid model. Granite 4 dense configs used to look like this."""
    fixture_path = FIXTURES_PATH / "ibm-granite" / "granite-4-8b-dense-hybrid" / "config.json"
    return _load_hf_config(fixture_path)


@pytest.fixture
def granite_4_hf_config():
    """Fixture providing real granite-4-8b-dense HF config."""
    fixture_path = FIXTURES_PATH / "ibm-granite" / "granite-4-8b-dense" / "config.json"
    return _load_hf_config(fixture_path)


@pytest.fixture
def embedding_hf_config():
    """Fixture providing real granite-embedding-125m-english HF config."""
    fixture_path = FIXTURES_PATH / "ibm-granite" / "granite-embedding-125m-english" / "config.json"
    return _load_hf_config(fixture_path)


@pytest.fixture
def micro_model_hf_config():
    """Fixture providing real micro-g3.3-8b-instruct-1b HF config."""
    fixture_path = FIXTURES_PATH / "ibm-ai-platform" / "micro-g3.3-8b-instruct-1b" / "config.json"
    return _load_hf_config(fixture_path)


def create_vllm_config(
    hf_config=None,
    world_size=1,
    max_model_len=None,
    max_num_seqs=None,
    max_num_batched_tokens=None,
    num_gpu_blocks_override=None,
    model_path=None,
):
    """Create a mock vllm_config for testing.

    Args:
        hf_config: HF config object (Mock or real)
        world_size: Tensor parallel size
        max_model_len: Maximum model length
        max_num_seqs: Max sequences (None for static batching)
        max_num_batched_tokens: Max batched tokens
        num_gpu_blocks_override: GPU blocks override value
        model_path: Model path string

    Returns:
        Mock vllm_config with specified attributes
    """
    vllm_config = Mock()

    # Model config
    model_config_attrs = {}
    if hf_config is not None:
        model_config_attrs["hf_config"] = hf_config
    if max_model_len is not None:
        model_config_attrs["max_model_len"] = max_model_len
    if model_path is not None:
        model_config_attrs["model"] = model_path
    vllm_config.model_config = Mock(**model_config_attrs)

    # Parallel config
    vllm_config.parallel_config = Mock(world_size=world_size)

    # Scheduler config
    vllm_config.scheduler_config = Mock(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
    )

    # Cache config
    vllm_config.cache_config = Mock(num_gpu_blocks_override=num_gpu_blocks_override)

    return vllm_config


# Made with Bob
