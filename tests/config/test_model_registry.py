"""Tests for ModelRegistry - registry operations and runtime matching."""

import pytest
import yaml
from unittest.mock import Mock

from sendnn_inference.config.model_config import ModelConfig
from sendnn_inference.config.model_registry import ModelConfigRegistry
from .conftest import create_vllm_config

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def static_batching_model():
    """Fixture providing a model config with static batching."""
    yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            batch_size: 4
          - prompt_len: 128
            batch_size: 2
          - prompt_len: 256
            batch_size: 1
    """
    data = yaml.safe_load(yaml_content)
    return ModelConfig.from_dict("test-model", data["models"]["test-model"])


@pytest.fixture
def mock_static_vllm_config():
    """Fixture providing a mock vllm config for static batching."""
    return create_vllm_config(
        hf_config=Mock(model_type="test"),
        world_size=1,
        max_num_seqs=None,
    )


class TestCBConfigMatchingLogic:
    """Tests for continuous batching config matching logic."""

    # NB: Using test_ prefix on some parameterize to not conflict with default
    # parameterization in ../conftest.py
    @pytest.mark.parametrize(
        "world_size,test_max_model_len,test_max_num_seqs,should_match,description",
        [
            (2, 8192, 32, False, "Should not match when only TP size differs"),
            (4, 4096, 32, False, "Should not match when only max_model_len differs"),
            (4, 8192, 16, False, "Should not match when only max_num_seqs differs"),
            (4, 8192, 32, True, "Should match when all parameters match"),
        ],
    )
    def test_cb_config_matching_any_mismatch_skips(
        self, world_size, test_max_model_len, test_max_num_seqs, should_match, description
    ):
        """Test that CB config is skipped if ANY parameter mismatches."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
        """
        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        registry.register_model(model_config)

        vllm_config = create_vllm_config(
            hf_config=Mock(model_type="test"),
            world_size=world_size,
            max_model_len=test_max_model_len,
            max_num_seqs=test_max_num_seqs,
        )
        configurator = registry.get_configurator_for_runtime(vllm_config)

        if should_match:
            assert configurator is not None, description
        else:
            assert configurator is None, description


class TestModelMatchingPriority:
    """Tests for model matching prioritization based on complexity."""

    def test_prioritizes_quantized_over_base_model(self):
        """Test that quantized model config is selected over base model when both match."""
        yaml_content = """
models:
  base-model:
    architecture:
      model_type: granite
      num_hidden_layers: 40
      hidden_size: 4096
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 4

  quantized-model:
    architecture:
      model_type: granite
      num_hidden_layers: 40
      hidden_size: 4096
      quantization_config:
        format: float-quantized
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 4
        """
        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()

        # Register both models
        base_config = ModelConfig.from_dict("base-model", data["models"]["base-model"])
        quant_config = ModelConfig.from_dict("quantized-model", data["models"]["quantized-model"])
        registry.register_model(base_config)
        registry.register_model(quant_config)

        # Create HF config with quantization
        vllm_model_config = Mock()
        vllm_model_config.model = "test/granite-fp8"
        vllm_model_config.hf_config = Mock(
            model_type="granite",
            num_hidden_layers=40,
            hidden_size=4096,
            quantization_config={"format": "float-quantized"},
        )

        # Should match the quantized model (higher complexity score)
        matched = registry.find_matching_model(vllm_model_config)
        assert matched is not None
        assert matched.name == "quantized-model"

    def test_base_model_matches_when_no_quantization(self):
        """Test that base model is selected when HF config has no quantization."""
        yaml_content = """
models:
  base-model:
    architecture:
      model_type: granite
      num_hidden_layers: 40
      hidden_size: 4096
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 4
  
  quantized-model:
    architecture:
      model_type: granite
      num_hidden_layers: 40
      hidden_size: 4096
      quantization_config:
        format: float-quantized
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 4
        """
        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()

        # Register both models
        base_config = ModelConfig.from_dict("base-model", data["models"]["base-model"])
        quant_config = ModelConfig.from_dict("quantized-model", data["models"]["quantized-model"])
        registry.register_model(base_config)
        registry.register_model(quant_config)

        # Create HF config WITHOUT quantization
        vllm_model_config = Mock()
        vllm_model_config.model = "test/granite-base"
        vllm_model_config.hf_config = Mock(
            model_type="granite",
            num_hidden_layers=40,
            hidden_size=4096,
        )

        # Should match the base model (quantized model won't match)
        matched = registry.find_matching_model(vllm_model_config)
        assert matched is not None
        assert matched.name == "base-model"

    def test_more_specific_attributes_win(self):
        """Test that patterns with more matching attributes are prioritized."""
        yaml_content = """
models:
  generic-model:
    architecture:
      model_type: granite
      num_hidden_layers: 40
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 4
  
  specific-model:
    architecture:
      model_type: granite
      num_hidden_layers: 40
      hidden_size: 4096
      vocab_size: 49159
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 4
        """
        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()

        # Register both models
        generic_config = ModelConfig.from_dict("generic-model", data["models"]["generic-model"])
        specific_config = ModelConfig.from_dict("specific-model", data["models"]["specific-model"])
        registry.register_model(generic_config)
        registry.register_model(specific_config)

        # Create HF config that matches both
        vllm_model_config = Mock()
        vllm_model_config.model = "test/granite"
        vllm_model_config.hf_config = Mock(
            model_type="granite",
            num_hidden_layers=40,
            hidden_size=4096,
            vocab_size=49159,
        )

        # Should match the more specific model (higher complexity score)
        matched = registry.find_matching_model(vllm_model_config)
        assert matched is not None
        assert matched.name == "specific-model"


class TestWarmupShapesSubset:
    """Tests for warmup shapes subset matching."""

    @pytest.mark.parametrize(
        "runtime_shapes,should_match,description",
        [
            ([(64, 4), (128, 2)], True, "Subset of shapes should match"),
            ([(256, 1)], True, "Single shape should match if in config"),
            ([(512, 1)], False, "Non-matching shape should not match"),
            ([(64, 4), (512, 1)], False, "Partial match should fail"),
            ([(256, 1), (64, 4)], True, "Different order should match"),
            ([], False, "Empty shapes should not match"),
        ],
        ids=["subset", "single", "no_match", "partial_fail", "order_independent", "empty"],
    )
    def test_warmup_shapes_matching(
        self,
        static_batching_model,
        mock_static_vllm_config,
        runtime_shapes,
        should_match,
        description,
    ):
        """Test various warmup shapes matching scenarios."""
        registry = ModelConfigRegistry()
        registry.register_model(static_batching_model)

        configurator = registry.get_configurator_for_runtime(
            mock_static_vllm_config, runtime_shapes
        )

        if should_match:
            assert configurator is not None, description
        else:
            assert configurator is None, description


class TestDuplicateRuntimeConfigDetection:
    """Tests for duplicate runtime configuration detection."""

    def test_registry_rejects_duplicate_cb_configs(self):
        """Test that registry detects and rejects duplicate CB configs."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
        """

        data = yaml.safe_load(yaml_content)

        with pytest.raises(ValueError, match="Duplicate runtime configuration"):
            ModelConfig.from_dict("test-model", data["models"]["test-model"])

    def test_registry_rejects_duplicate_static_configs(self):
        """Test that registry detects and rejects duplicate static batching configs."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            new_tokens: 20
            batch_size: 4
          - prompt_len: 128
            new_tokens: 40
            batch_size: 2
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            new_tokens: 20
            batch_size: 4
          - prompt_len: 128
            new_tokens: 40
            batch_size: 2
        """

        data = yaml.safe_load(yaml_content)

        with pytest.raises(ValueError, match="Duplicate runtime configuration"):
            ModelConfig.from_dict("test-model", data["models"]["test-model"])

    def test_registry_rejects_duplicate_static_configs_different_order(self):
        """Test that duplicate detection works even with different warmup shape order."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            batch_size: 4
          - prompt_len: 128
            batch_size: 2
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 128
            batch_size: 2
          - prompt_len: 64
            batch_size: 4
        """

        data = yaml.safe_load(yaml_content)

        with pytest.raises(ValueError, match="Duplicate runtime configuration"):
            ModelConfig.from_dict("test-model", data["models"]["test-model"])

    def test_registry_allows_different_cb_configs_same_tp(self):
        """Test that different CB configs with same TP are allowed."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
      - tp_size: 4
        max_model_len: 16384
        max_num_seqs: 16
        """

        data = yaml.safe_load(yaml_content)

        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        assert len(model_config.continuous_batching_configs) == 2

    def test_registry_allows_different_static_configs_same_tp(self):
        """Test that different static configs with same TP are allowed."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            new_tokens: 20
            batch_size: 4
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 128
            new_tokens: 40
            batch_size: 2
        """

        data = yaml.safe_load(yaml_content)

        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        assert len(model_config.static_batching_configs) == 2

    def test_warmup_shapes_only_match_static_batching(self):
        """Test that providing warmup_shapes only matches static batching configs.

        This test verifies that when warmup_shapes are provided, the registry
        prioritizes static batching configs over continuous batching configs,
        even if a CB config exists with matching TP size.
        """
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 32
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            batch_size: 4
        """
        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        registry.register_model(model_config)

        # Create vllm config that would match the CB config
        vllm_config = create_vllm_config(
            hf_config=Mock(model_type="test"),
            world_size=1,
            max_model_len=8192,
            max_num_seqs=32,
        )

        # Provide warmup_shapes that DON'T match the static batching config
        non_matching_warmup_shapes = [(128, 2)]  # Not in static config

        configurator = registry.get_configurator_for_runtime(
            vllm_config, warmup_shapes=non_matching_warmup_shapes
        )

        # Should return None because warmup_shapes were provided but don't match
        # the static batching config. The CB config should be ignored when
        # warmup_shapes are provided.
        assert configurator is None, (
            "When warmup_shapes are provided, only static batching configs should be "
            "considered, even if a CB config matches the runtime parameters"
        )

    def test_warmup_shapes_match_static_batching_ignores_cb(self):
        """Test that matching warmup_shapes returns static config, ignoring CB config.

        This test verifies that when warmup_shapes match a static batching config,
        that config is used even if a CB config also matches the runtime parameters.
        """
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 32
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 64
            batch_size: 4
          - prompt_len: 128
            batch_size: 2
        """
        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        registry.register_model(model_config)

        # Create vllm config that would match the CB config
        vllm_config = create_vllm_config(
            hf_config=Mock(model_type="test"),
            world_size=1,
            max_model_len=8192,
            max_num_seqs=32,
        )

        # Provide warmup_shapes that DO match the static batching config
        matching_warmup_shapes = [(64, 4)]

        configurator = registry.get_configurator_for_runtime(
            vllm_config, warmup_shapes=matching_warmup_shapes
        )

        # Should return a configurator with no device_config (static batching)
        assert configurator is not None, (
            "When warmup_shapes match a static batching config, a configurator should be returned"
        )
        assert configurator.device_config is None, (
            "Static batching configs should not have device_config"
        )


# Made with Bob
