"""Integration tests for the configuration system - end-to-end scenarios."""

import os
from unittest.mock import Mock, patch

import pytest

from sendnn_inference.config.configurators.model_configurator import ModelConfigurator
from .conftest import create_vllm_config

pytestmark = pytest.mark.skip_global_cleanup


class TestRegistryLoading:
    """Tests for registry initialization and model loading."""

    def test_load_real_config_file(self, registry):
        """Test loading the actual model_configs.yaml file."""
        models = registry.list_models()

        # Should have at least 9 models from the config file
        # (may have more if additional models are added)
        assert len(models) >= 9

        # Verify specific models are present
        expected_models = [
            "ibm-granite/granite-3.3-8b-instruct",
            "ibm-granite/granite-3.3-8b-instruct-FP8",
            "ibm-granite/granite-4-8b-dense",
            "ibm-granite/granite-embedding-125m-english",
        ]
        for model in expected_models:
            assert model in models, f"Expected model {model} missing"


class TestModelMatching:
    """Tests for model detection and matching."""

    def test_match_granite_3_3_cb_config(self, registry, granite_3_3_hf_config):
        """Test matching granite-3.3-8b-instruct with CB config and getting configurator."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-3.3-8b-instruct"
        assert configurator.device_config is not None
        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT" in configurator.device_config.env_vars

    def test_match_granite_4_dense_hybrid_config(self, registry, granite_4_hf_dense_hybrid_config):
        """Test matching granite-4-8b-dense configs that have type granitemoehybrid."""
        vllm_config = create_vllm_config(
            hf_config=granite_4_hf_dense_hybrid_config,
            world_size=4,
            max_model_len=32768,
            max_num_seqs=32,
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        # This is really a dense model, but it has model type "granitemoehybrid"
        # It has the same overrides as the regular dense variant
        assert configurator.model_config.name == "ibm-granite/granite-4-8b-dense-hybrid"
        assert configurator.device_config is not None

    def test_match_granite_4_dense_config(self, registry, granite_4_hf_config):
        """Test matching granite-4-8b-dense configs that aren't spoofed moe hybrid models."""
        vllm_config = create_vllm_config(
            hf_config=granite_4_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-4-8b-dense"
        assert configurator.device_config is not None

    def test_embedding_models_have_no_device_configs(self, registry, embedding_hf_config):
        """Test that embedding models don't have device_configs."""
        vllm_config = create_vllm_config(
            hf_config=embedding_hf_config,
            world_size=1,
            max_num_seqs=None,  # Static batching
        )

        warmup_shapes = [(512, 64)]
        configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

        assert configurator is not None
        assert configurator.model_config.name == "ibm-granite/granite-embedding-125m-english"
        assert configurator.device_config is None

    def test_generation_models_with_tp4_have_device_configs(self, registry, granite_3_3_hf_config):
        """Test that generation models with TP=4 have device_configs."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert configurator.device_config is not None
        assert configurator.device_config.env_vars is not None
        assert configurator.device_config.num_gpu_blocks_override is not None


class TestUnregisteredModels:
    """Tests for unregistered model handling."""

    def test_unregistered_model_returns_none(self, registry):
        """Test that unregistered model type returns None from registry (no error)."""
        hf_config = Mock(model_type="unknown_model_type")
        vllm_config = create_vllm_config(
            hf_config=hf_config,
            max_model_len=8192,
            world_size=1,
            max_num_seqs=4,
        )

        # Should return None, not raise error
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

    def test_micro_model_not_in_registry(self, registry, micro_model_hf_config):
        """Test that micro model (not in registry) returns None but doesn't error."""
        vllm_config = create_vllm_config(
            hf_config=micro_model_hf_config,
            max_model_len=8192,
            world_size=1,
            max_num_seqs=4,
        )

        # Should return None (micro model not in registry)
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

    def test_log_message_when_model_not_found(self, registry, caplog_sendnn_inference):
        """Test that appropriate message is logged when model not found."""
        hf_config = Mock(model_type="unknown_model")
        vllm_config = create_vllm_config(
            hf_config=hf_config,
            max_model_len=8192,
            model_path="unknown-model",
            world_size=1,
            max_num_seqs=4,
        )

        # Try to get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

        # Check that debug message was logged
        assert any(
            "No matching model configuration found" in record.message
            for record in caplog_sendnn_inference.records
        )


class TestGraniteGPUBlocksOverrides:
    """Tests for GPU blocks overrides for Granite models."""

    @pytest.mark.cpu
    @pytest.mark.parametrize(
        "hf_config_fixture, expected_blocks",
        [
            ("granite_3_3_hf_config", 8192),
            ("granite_4_hf_config", 8192),
        ],
        ids=[
            "g3.3_default",
            "g4_default",
        ],
    )
    def test_granite_gpu_blocks_overrides(
        self,
        request,
        registry,
        hf_config_fixture,
        expected_blocks,
    ):
        """Test GPU blocks and env var overrides for granite models."""

        # Get the HF config from the fixture
        hf_config = request.getfixturevalue(hf_config_fixture)

        # Must ensure no env vars have been overridden before testing
        with patch.dict(os.environ, clear=True):
            # Create vllm_config for CB with TP=4 using helper
            granite_config = create_vllm_config(
                hf_config=hf_config,
                world_size=4,
                max_model_len=32768,
                max_num_seqs=32,
            )

            # Get configurator and apply configuration
            configurator = registry.get_configurator_for_runtime(granite_config)
            assert configurator is not None, (
                f"Model with fixture {hf_config_fixture} should have a matching configurator"
            )

            configurator.configure(granite_config)

            # Verify the configuration was applied correctly
            assert granite_config.cache_config.num_gpu_blocks_override == expected_blocks

            # Verify environment variables were set
            tkv_limit = os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT")
            assert tkv_limit is not None, "VLLM_DT_MAX_BATCH_TKV_LIMIT should be set"
            assert int(tkv_limit) == 128 * 1024

            hdma_size = os.getenv("FLEX_HDMA_P2PSIZE")
            assert hdma_size is not None, "FLEX_HDMA_P2PSIZE should be set"
            assert int(hdma_size) == 256 * 1024 * 1024


# Made with Bob
