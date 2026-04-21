"""Tests for error handling in the configuration system."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from sendnn_inference.config.model_registry import ModelConfigRegistry


@pytest.fixture
def test_config_path():
    """Provide path to test-specific model configuration YAML."""
    return Path(__file__).parent / "fixtures" / "test_error_handling_models.yaml"


@pytest.fixture
def test_registry(test_config_path):
    """Provide a registry initialized with test configuration."""
    registry = ModelConfigRegistry()
    registry.initialize(test_config_path)
    return registry


class TestRegistryErrorHandling:
    """Tests for registry error handling."""

    def test_initialize_with_nonexistent_file(self, caplog_sendnn_inference):
        """Test that registry handles nonexistent config file gracefully."""
        registry = ModelConfigRegistry()
        nonexistent_path = Path("/nonexistent/path/to/config.yaml")

        with pytest.raises(FileNotFoundError, match="Model configuration file not found"):
            registry.initialize(nonexistent_path)

    def test_initialize_with_invalid_yaml(self):
        """Test that registry handles invalid YAML gracefully."""
        registry = ModelConfigRegistry()

        # Create temp file with invalid YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken")
            temp_path = Path(f.name)

        try:
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Failed to load model configurations"):
                registry.initialize(temp_path)
        finally:
            temp_path.unlink()

    def test_initialize_twice_skips_second_load(self, test_config_path, caplog_sendnn_inference):
        """Test that initializing twice doesn't reload."""
        registry = ModelConfigRegistry()

        # First initialization
        registry.initialize(test_config_path)
        model_count_first = len(registry.list_models())

        # Second initialization should skip
        registry.initialize(test_config_path)
        model_count_second = len(registry.list_models())

        # Should have same count
        assert model_count_first == model_count_second

        # Should log debug message about skipping
        assert any(
            "already initialized" in record.message for record in caplog_sendnn_inference.records
        )

    def test_find_matching_model_with_no_hf_config(self, test_registry, caplog_sendnn_inference):
        """Test matching when vllm_config has no HF config."""
        registry = test_registry

        # Create mock vllm_config with no HF config
        vllm_model_config = Mock()
        vllm_model_config.hf_config = None
        vllm_model_config.model = "some-model"

        # Should return None
        matched = registry.find_matching_model(vllm_model_config)
        assert matched is None

        # Should log debug message
        assert any(
            "No HF config available" in record.message for record in caplog_sendnn_inference.records
        )

    def test_get_configurator_for_runtime_with_no_model_match(
        self, test_registry, caplog_sendnn_inference
    ):
        """Test getting configurator when model doesn't match."""
        registry = test_registry

        # Create mock vllm_config with unknown model
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=Mock(model_type="unknown_type"), model="unknown")
        vllm_config.parallel_config = Mock(world_size=1)
        vllm_config.scheduler_config = Mock(max_num_seqs=4)

        # Should return None
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

        # Should log debug message
        assert any(
            "No model architecture match" in record.message
            for record in caplog_sendnn_inference.records
        )

    def test_get_configurator_for_runtime_with_unsupported_runtime_config(
        self, test_registry, caplog_sendnn_inference
    ):
        """Test getting configurator when model matches but runtime config doesn't."""
        registry = test_registry

        # Create HF config matching our test model architecture
        hf_config = Mock()
        hf_config.model_type = "granite"
        hf_config.num_hidden_layers = 40
        hf_config.max_position_embeddings = 131072
        hf_config.hidden_size = 4096
        hf_config.vocab_size = 49159
        hf_config.num_key_value_heads = 8
        hf_config.num_attention_heads = 32

        # Create vllm_config with unsupported runtime params (TP=8, which doesn't exist in config)
        vllm_config = Mock()
        vllm_config.model_config = Mock(
            hf_config=hf_config, max_model_len=32768, model="test-granite-model"
        )
        vllm_config.parallel_config = Mock(world_size=8)  # TP=8 not in config
        vllm_config.scheduler_config = Mock(max_num_seqs=32)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Should return None
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

        # Should log warning about unsupported runtime config
        assert any(
            "does not support the requested runtime" in record.message
            for record in caplog_sendnn_inference.records
        )

    def test_empty_yaml_file(self):
        """Test loading an empty YAML file."""
        registry = ModelConfigRegistry()

        # Create temp file with empty YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            # Should not raise, just load 0 models
            registry.initialize(temp_path)
            assert len(registry.list_models()) == 0
        finally:
            temp_path.unlink()

    def test_yaml_without_models_key(self):
        """Test loading YAML without 'models' key."""
        registry = ModelConfigRegistry()

        # Create temp file with YAML but no models key
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"other_key": "value"}, f)
            temp_path = Path(f.name)

        try:
            # Should not raise, just load 0 models
            registry.initialize(temp_path)
            assert len(registry.list_models()) == 0
        finally:
            temp_path.unlink()


class TestConfiguratorEdgeCases:
    """Tests for configurator edge cases."""

    def test_configure_with_none_device_config(self):
        """Test that configurator handles None device_config gracefully."""
        from sendnn_inference.config.model_config import ModelConfig, ArchitecturePattern
        from sendnn_inference.config.configurators.model_configurator import ModelConfigurator

        # Create minimal model config
        model_config = ModelConfig(
            name="test-model",
            architecture=ArchitecturePattern(model_name="test-model", model_type="test"),
            continuous_batching_configs=[],
            static_batching_configs=[],
        )

        # Create configurator with no device config
        configurator = ModelConfigurator(model_config, device_config=None)

        # Create mock vllm_config
        vllm_config = Mock()
        vllm_config.parallel_config = Mock(world_size=1)
        vllm_config.scheduler_config = Mock(max_num_seqs=4, max_num_batched_tokens=None)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Should return empty summary
        summary = configurator.configure(vllm_config)

        assert summary.model_name == "test-model"
        assert summary.tp_size == 1
        assert len(summary.env_vars) == 0
        assert summary.num_blocks is None

    def test_registry_os_error_path(self):
        """Test that OSError is caught and re-raised as RuntimeError when file is unreadable."""
        from sendnn_inference.config.model_registry import ModelConfigRegistry
        from pathlib import Path
        import tempfile
        import os

        registry = ModelConfigRegistry()

        # Create a file with no read permissions to trigger OSError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("models: {}")
            temp_path = Path(f.name)

        try:
            # Remove read permissions
            os.chmod(temp_path, 0o000)

            # Should raise RuntimeError wrapping OSError
            with pytest.raises(RuntimeError, match="Failed to load model configurations"):
                registry.initialize(temp_path)
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            temp_path.unlink()


# Made with Bob
