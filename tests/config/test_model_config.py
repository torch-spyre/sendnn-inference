"""Tests for model configuration data structures."""

import pytest

from sendnn_inference.config.model_config import (
    ArchitecturePattern,
    ContinuousBatchingConfig,
    DeviceConfig,
    ModelConfig,
    StaticBatchingConfig,
    WarmupShape,
)

pytestmark = pytest.mark.skip_global_cleanup


class TestArchitecturePattern:
    """Tests for ArchitecturePattern dataclass."""

    def test_create_minimal_pattern(self):
        """Test creating pattern with only required fields."""
        pattern = ArchitecturePattern(model_name="test-model", model_type="llama")
        assert pattern.model_name == "test-model"
        assert pattern.model_type == "llama"
        assert pattern.attributes == {}

    def test_create_full_pattern(self):
        """Test creating pattern with all fields."""
        quant_config = {"format": "float-quantized"}
        attributes = {
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "hidden_size": 4096,
            "vocab_size": 49152,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "num_experts_per_tok": 2,
            "quantization_config": quant_config,
        }
        pattern = ArchitecturePattern(
            model_name="granite-model",
            model_type="granite",
            attributes=attributes,
        )
        assert pattern.model_name == "granite-model"
        assert pattern.model_type == "granite"
        assert pattern.attributes["num_hidden_layers"] == 32
        assert pattern.attributes["max_position_embeddings"] == 4096
        assert pattern.attributes["hidden_size"] == 4096
        assert pattern.attributes["vocab_size"] == 49152
        assert pattern.attributes["num_key_value_heads"] == 8
        assert pattern.attributes["num_attention_heads"] == 32
        assert pattern.attributes["num_experts_per_tok"] == 2
        assert pattern.attributes["quantization_config"] == quant_config

    def test_from_dict_minimal(self):
        """Test creating pattern from minimal dict."""
        data = {"model_type": "roberta"}
        pattern = ArchitecturePattern.from_dict("roberta-model", data)
        assert pattern.model_name == "roberta-model"
        assert pattern.model_type == "roberta"
        assert pattern.attributes == {}

    def test_from_dict_full(self):
        """Test creating pattern from complete dict."""
        data = {
            "model_type": "granite",
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "hidden_size": 4096,
            "vocab_size": 49152,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "num_experts_per_tok": 2,
            "quantization_config": {"format": "float-quantized"},
        }
        pattern = ArchitecturePattern.from_dict("granite-model", data)
        assert pattern.model_name == "granite-model"
        assert pattern.model_type == "granite"
        assert pattern.attributes["num_hidden_layers"] == 32
        assert pattern.attributes["max_position_embeddings"] == 4096
        assert pattern.attributes["hidden_size"] == 4096
        assert pattern.attributes["vocab_size"] == 49152
        assert pattern.attributes["num_key_value_heads"] == 8
        assert pattern.attributes["num_attention_heads"] == 32
        assert pattern.attributes["num_experts_per_tok"] == 2
        assert pattern.attributes["quantization_config"] == {"format": "float-quantized"}

    def test_field_count_minimal_pattern(self):
        """Test field count for minimal pattern (no attributes)."""
        pattern = ArchitecturePattern(model_name="test-model", model_type="llama")
        assert pattern.field_count == 0

    def test_field_count_with_attributes(self):
        """Test field count increases with attributes."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={
                "num_hidden_layers": 32,
                "hidden_size": 4096,
            },
        )
        assert pattern.field_count == 2

    def test_field_count_with_quantization_config(self):
        """Test field count counts quantization_config and its nested fields."""
        pattern = ArchitecturePattern(
            model_name="test-model-fp8",
            model_type="granite",
            attributes={
                "num_hidden_layers": 32,
                "quantization_config": {
                    "format": "float-quantized",
                    "quant_method": "fp8",
                },
            },
        )
        # 1 for num_hidden_layers + 1 for quantization_config + 2 for keys in quantization_config
        assert pattern.field_count == 4

    def test_from_dict_rejects_none_values(self):
        """Test that from_dict rejects None attribute values."""
        data = {
            "model_type": "granite",
            "num_hidden_layers": 32,
            "hidden_size": None,  # Should be rejected
        }
        with pytest.raises(ValueError, match="cannot be None"):
            ArchitecturePattern.from_dict("test-model", data)

    def test_from_dict_rejects_nested_none_values(self):
        """Test that from_dict rejects None values in nested dicts."""
        data = {
            "model_type": "granite",
            "quantization_config": {
                "format": "float-quantized",
                "quant_method": None,  # Should be rejected
            },
        }
        with pytest.raises(ValueError, match="cannot be None"):
            ArchitecturePattern.from_dict("test-model", data)


class TestDeviceConfig:
    """Tests for DeviceConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating device config with only required fields."""
        config = DeviceConfig(tp_size=1)
        assert config.tp_size == 1
        assert config.env_vars == {}
        assert config.num_gpu_blocks_override is None

    def test_create_with_env_vars(self):
        """Test creating device config with environment variables."""
        env_vars = {"SOME_ENV_VAR": "some_value"}
        config = DeviceConfig(tp_size=2, env_vars=env_vars)
        assert config.tp_size == 2
        assert config.env_vars == env_vars

    def test_create_with_gpu_blocks_int(self):
        """Test creating device config with integer GPU blocks override."""
        config = DeviceConfig(tp_size=1, num_gpu_blocks_override=1000)
        assert config.num_gpu_blocks_override == 1000

    def test_from_dict_minimal(self):
        """Test creating device config from minimal dict."""
        config = DeviceConfig.from_dict(tp_size=1, data={})
        assert config.tp_size == 1
        assert config.env_vars == {}

    def test_from_dict_full(self):
        """Test creating device config from complete dict."""
        data = {
            "env_vars": {"SOME_ENV_VAR": "some_value"},
            "num_gpu_blocks_override": 1000,
        }
        config = DeviceConfig.from_dict(tp_size=2, data=data)
        assert config.tp_size == 2
        assert config.env_vars == data["env_vars"]
        assert config.num_gpu_blocks_override == data["num_gpu_blocks_override"]


class TestWarmupShape:
    """Tests for WarmupShape dataclass."""

    def test_create_warmup_shape(self):
        """Test creating warmup shape."""
        shape = WarmupShape(prompt_len=64, batch_size=4)
        assert shape.prompt_len == 64
        assert shape.batch_size == 4

    def test_to_tuple(self):
        """Test converting warmup shape to tuple."""
        shape = WarmupShape(prompt_len=64, batch_size=4)
        assert shape.to_tuple() == (64, 4)

    def test_from_dict(self):
        """Test creating warmup shape from dict."""
        data = {"prompt_len": 128, "batch_size": 2}
        shape = WarmupShape.from_dict(data)
        assert shape.prompt_len == 128
        assert shape.batch_size == 2

    def test_from_dict_missing_key(self):
        """Test that missing keys raise ValueError."""
        data = {"prompt_len": 128}
        with pytest.raises(ValueError, match="Missing key"):
            WarmupShape.from_dict(data)

    def test_from_dict_invalid_value(self):
        """Test that invalid values raise ValueError."""
        data = {"prompt_len": "not_an_int", "batch_size": 2}
        with pytest.raises(ValueError, match="must be valid integers"):
            WarmupShape.from_dict(data)


class TestStaticBatchingConfig:
    """Tests for StaticBatchingConfig dataclass."""

    def test_create_config(self):
        """Test creating static batching config."""
        warmup_shapes = [
            WarmupShape(prompt_len=64, batch_size=4),
            WarmupShape(prompt_len=128, batch_size=2),
        ]
        config = StaticBatchingConfig(tp_size=1, warmup_shapes=warmup_shapes)
        assert config.tp_size == 1
        assert len(config.warmup_shapes) == 2
        assert config.warmup_shapes[0].to_tuple() == (64, 4)
        assert config.warmup_shapes[1].to_tuple() == (128, 2)

    def test_from_dict(self):
        """Test creating static batching config from dict."""
        data = {
            "tp_size": 1,
            "warmup_shapes": [
                {"prompt_len": 64, "batch_size": 4},
                {"prompt_len": 128, "batch_size": 2},
            ],
        }
        config = StaticBatchingConfig.from_dict(data)
        assert config.tp_size == 1
        assert len(config.warmup_shapes) == 2
        assert config.warmup_shapes[0].to_tuple() == (64, 4)
        assert config.warmup_shapes[1].to_tuple() == (128, 2)


class TestContinuousBatchingConfig:
    """Tests for ContinuousBatchingConfig dataclass."""

    def test_create_config(self):
        """Test creating continuous batching config."""
        config = ContinuousBatchingConfig(tp_size=1, max_model_len=2048, max_num_seqs=256)
        assert config.tp_size == 1
        assert config.max_model_len == 2048
        assert config.max_num_seqs == 256
        assert config.device_config is None

    def test_create_config_with_device_config(self):
        """Test creating continuous batching config with device config."""
        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        config = ContinuousBatchingConfig(
            tp_size=4, max_model_len=32768, max_num_seqs=32, device_config=device_config
        )
        assert config.tp_size == 4
        assert config.max_model_len == 32768
        assert config.max_num_seqs == 32
        assert config.device_config is not None
        assert config.device_config.env_vars["TEST_VAR"] == "123"

    def test_from_dict(self):
        """Test creating continuous batching config from dict."""
        data = {
            "tp_size": 2,
            "max_model_len": 4096,
            "max_num_seqs": 128,
        }
        config = ContinuousBatchingConfig.from_dict(data)
        assert config.tp_size == 2
        assert config.max_model_len == 4096
        assert config.max_num_seqs == 128
        assert config.device_config is None

    def test_from_dict_with_device_config(self):
        """Test CB config with nested device config."""
        data = {
            "tp_size": 4,
            "max_model_len": 32768,
            "max_num_seqs": 32,
            "device_config": {
                "env_vars": {"TEST_VAR": "123"},
                "num_gpu_blocks_override": 8192,
            },
        }

        cb_config = ContinuousBatchingConfig.from_dict(data)

        assert cb_config.tp_size == 4
        assert cb_config.max_model_len == 32768
        assert cb_config.max_num_seqs == 32
        assert cb_config.device_config is not None
        assert cb_config.device_config.env_vars["TEST_VAR"] == "123"
        assert cb_config.device_config.num_gpu_blocks_override == 8192


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating model config with minimal fields."""
        architecture = ArchitecturePattern(model_name="test-model", model_type="llama")
        config = ModelConfig(name="test-model", architecture=architecture)
        assert config.name == "test-model"
        assert config.architecture == architecture
        assert config.static_batching_configs == []
        assert config.continuous_batching_configs == []

    def test_create_with_configs(self):
        """Test creating model config with configurations."""
        architecture = ArchitecturePattern(model_name="granite-model", model_type="granite")
        warmup_shape = WarmupShape(prompt_len=64, batch_size=4)
        static_config = StaticBatchingConfig(tp_size=1, warmup_shapes=[warmup_shape])
        device_config = DeviceConfig(tp_size=4, env_vars={"TEST": "value"})
        cb_config = ContinuousBatchingConfig(
            tp_size=4, max_model_len=2048, max_num_seqs=256, device_config=device_config
        )

        config = ModelConfig(
            name="granite-model",
            architecture=architecture,
            static_batching_configs=[static_config],
            continuous_batching_configs=[cb_config],
        )
        assert config.name == "granite-model"
        assert len(config.static_batching_configs) == 1
        assert len(config.continuous_batching_configs) == 1
        assert cb_config.device_config == device_config

    def test_from_dict_minimal(self):
        """Test creating model config from minimal dict with at least one config."""
        data = {
            "architecture": {"model_type": "llama"},
            "static_batching_configs": [
                {
                    "tp_size": 1,
                    "warmup_shapes": [{"prompt_len": 64, "batch_size": 4}],
                }
            ],
            "continuous_batching_configs": [],
        }
        config = ModelConfig.from_dict(name="test-model", data=data)
        assert config.name == "test-model"
        assert config.architecture.model_type == "llama"
        assert config.architecture.model_name == "test-model"
        assert len(config.static_batching_configs) == 1
        assert config.continuous_batching_configs == []

    def test_from_dict_no_configs_raises_error(self):
        """Test that creating model config with no runtime configs raises ValueError."""
        data = {
            "architecture": {"model_type": "llama"},
            "static_batching_configs": [],
            "continuous_batching_configs": [],
        }
        with pytest.raises(ValueError, match="must have at least one runtime configuration"):
            ModelConfig.from_dict(name="test-model", data=data)

    def test_from_dict_with_new_config_format(self):
        """Test creating model config from dict with new format."""
        data = {
            "architecture": {"model_type": "granite"},
            "static_batching_configs": [
                {
                    "tp_size": 1,
                    "warmup_shapes": [{"prompt_len": 64, "batch_size": 4}],
                }
            ],
            "continuous_batching_configs": [
                {"tp_size": 1, "max_model_len": 2048, "max_num_seqs": 256}
            ],
        }
        config = ModelConfig.from_dict(name="granite-model", data=data)
        assert len(config.static_batching_configs) == 1
        assert len(config.continuous_batching_configs) == 1
        assert config.static_batching_configs[0].tp_size == 1
        assert config.static_batching_configs[0].warmup_shapes[0].to_tuple() == (64, 4)
        assert config.continuous_batching_configs[0].tp_size == 1
        assert config.continuous_batching_configs[0].max_model_len == 2048

    def test_from_dict_with_nested_device_config(self):
        """Test creating model config with nested device config in CB config."""
        data = {
            "architecture": {"model_type": "granite"},
            "static_batching_configs": [
                {
                    "tp_size": 1,
                    "warmup_shapes": [{"prompt_len": 64, "batch_size": 4}],
                }
            ],
            "continuous_batching_configs": [
                {
                    "tp_size": 4,
                    "max_model_len": 32768,
                    "max_num_seqs": 32,
                    "device_config": {
                        "env_vars": {"VLLM_DT_MAX_BATCH_TKV_LIMIT": "131072"},
                        "num_gpu_blocks_override": 8192,
                    },
                }
            ],
        }
        config = ModelConfig.from_dict(name="granite-model", data=data)
        assert len(config.continuous_batching_configs) == 1
        cb_config = config.continuous_batching_configs[0]
        assert cb_config.tp_size == 4
        assert cb_config.device_config is not None
        assert cb_config.device_config.env_vars["VLLM_DT_MAX_BATCH_TKV_LIMIT"] == "131072"
        assert cb_config.device_config.num_gpu_blocks_override == 8192

    def test_model_config_no_device_configs_field(self):
        """Test that ModelConfig no longer has device_configs field."""
        data = {
            "architecture": {"model_type": "test"},
            "static_batching_configs": [
                {
                    "tp_size": 1,
                    "warmup_shapes": [{"prompt_len": 512, "batch_size": 64}],
                }
            ],
            "continuous_batching_configs": [
                {"tp_size": 1, "max_model_len": 1000, "max_num_seqs": 10}
            ],
        }

        model_config = ModelConfig.from_dict("test-model", data)

        # Verify device_configs field doesn't exist
        assert not hasattr(model_config, "device_configs")
        assert len(model_config.static_batching_configs) == 1
        assert len(model_config.continuous_batching_configs) == 1


class TestYAMLAnchorResolution:
    """Tests for YAML anchor resolution in device configs."""

    def test_yaml_anchor_resolution(self):
        """Test that YAML anchors are resolved correctly."""
        import yaml

        yaml_content = """
device_config_templates:
  test_config: &test_anchor
    env_vars:
      TEST_VAR: "123"
    num_gpu_blocks_override: 8192

models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 1000
        max_num_seqs: 10
        device_config: *test_anchor
        """

        data = yaml.safe_load(yaml_content)
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])

        assert len(model_config.continuous_batching_configs) == 1
        cb_config = model_config.continuous_batching_configs[0]
        assert cb_config.device_config is not None
        assert cb_config.device_config.env_vars["TEST_VAR"] == "123"
        assert cb_config.device_config.num_gpu_blocks_override == 8192


# Made with Bob
