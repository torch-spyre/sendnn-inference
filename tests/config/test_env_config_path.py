"""Tests for environment variable configuration of model_configs.yaml path."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from sendnn_inference import envs
from sendnn_inference.config.model_registry import ModelConfigRegistry


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
models:
  env-test-model:
    architecture:
      model_type: test_env
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 4096
        max_num_seqs: 16
""")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry singleton between tests."""
    ModelConfigRegistry._instance = None
    ModelConfigRegistry._initialized = False
    envs.clear_env_cache()
    yield
    ModelConfigRegistry._instance = None
    ModelConfigRegistry._initialized = False
    envs.clear_env_cache()


class TestEnvVarConfigPath:
    """Tests for SENDNN_INFERENCE_MODEL_CONFIG_FILE environment variable."""

    def test_env_var_overrides_default(self, temp_config_file):
        """Test that env var is used when no explicit path provided."""
        with patch.dict(os.environ, {"SENDNN_INFERENCE_MODEL_CONFIG_FILE": temp_config_file}):
            envs.clear_env_cache()
            registry = ModelConfigRegistry.get_instance()
            registry.initialize()

            assert "env-test-model" in registry.list_models()

    def test_explicit_path_takes_precedence(self, temp_config_file):
        """Test priority order: explicit path > env var > default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
models:
  explicit-model:
    architecture:
      model_type: test_explicit
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 32
""")
            explicit_path = f.name

        try:
            with patch.dict(os.environ, {"SENDNN_INFERENCE_MODEL_CONFIG_FILE": temp_config_file}):
                envs.clear_env_cache()
                registry = ModelConfigRegistry.get_instance()
                registry.initialize(config_path=Path(explicit_path))

                models = registry.list_models()
                assert "explicit-model" in models
                assert "env-test-model" not in models
        finally:
            os.unlink(explicit_path)

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with patch.dict(
            os.environ, {"SENDNN_INFERENCE_MODEL_CONFIG_FILE": "/tmp/nonexistent.yaml"}
        ):
            envs.clear_env_cache()
            registry = ModelConfigRegistry.get_instance()

            with pytest.raises(FileNotFoundError, match="Model configuration file not found"):
                registry.initialize()

    def test_empty_file_creates_empty_registry(self):
        """Test that empty YAML file results in empty registry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            empty_path = f.name

        try:
            with patch.dict(os.environ, {"SENDNN_INFERENCE_MODEL_CONFIG_FILE": empty_path}):
                envs.clear_env_cache()
                registry = ModelConfigRegistry.get_instance()
                registry.initialize()

                assert len(registry.list_models()) == 0
        finally:
            os.unlink(empty_path)


# Made with Bob
