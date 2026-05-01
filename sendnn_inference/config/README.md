# Model Configuration System

This directory contains the model configuration system for SenDNN Inference,
which provides a clean, extensible way to manage model-specific configurations.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Overview](#overview)
- [Architecture](#architecture)
- [Adding a New Model](#adding-a-new-model)
- [Configuration Schema](#configuration-schema)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Benefits](#benefits)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Directory Structure

```text
sendnn_inference/config/
├── __init__.py
├── model_config.py          # Data structures
├── model_configs.yaml       # Model definitions
├── model_matcher.py         # Pattern matching
├── model_registry.py        # Registry singleton
├── README.md
└── configurators/
    ├── __init__.py
    └── model_configurator.py  # Configuration logic
```

## Quick Start

To use the configuration system:

```python
from sendnn_inference.config.model_registry import get_model_registry

# Get the registry (auto-initializes with default or env var path)
registry = get_model_registry()

# Get configurator for your runtime
#   warmup_shapes is None for continuous batching
configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

# Apply configuration
if configurator:
    summary = configurator.configure(vllm_config)
    logger.info(summary.format_log_message())
```

### Custom Configuration File

By default, the registry loads from `sendnn_inference/config/model_configs.yaml`. You
can override this in three ways (in priority order):

1. **Explicit path**: Pass `config_path` to `initialize()`

   ```python
   registry.initialize(config_path=Path("/path/to/custom_config.yaml"))
   ```

2. **Environment variable**: Set `SENDNN_INFERENCE_MODEL_CONFIG_FILE`

   ```bash
   export SENDNN_INFERENCE_MODEL_CONFIG_FILE=/path/to/custom_config.yaml
   ```

3. **Default**: Uses built-in `model_configs.yaml`

## Overview

Uses a declarative YAML-based approach, making it easy to add new models and
maintain existing configurations.

### Benefits

1. **Extensibility**: Add new models by editing YAML only
2. **Maintainability**: Centralized configuration, no scattered code
3. **Testability**: Easy to test configurations in isolation
4. **Documentation**: YAML serves as self-documenting configuration
5. **Simplicity**: Single configurator handles all models
6. **Flexibility**: Configurations can be omitted

### Key Components

1. **`model_configs.yaml`**: Declarative model definitions with architecture
   patterns, runtime configs, and device configurations
2. **`model_registry.py`**: Singleton registry that loads and manages model configurations
3. **`model_matcher.py`**: Pattern-based matching to identify models from
   HuggingFace configs
4. **`model_config.py`**: Data structures (ModelConfig, ArchitecturePattern,
   DeviceConfig, RuntimeConfig)
5. **`configurators/model_configurator.py`**: Universal configurator for most models

## Architecture

```text
┌─────────────────────┐
│  model_configs.yaml │  ← Declarative configuration
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ModelConfigRegistry│  ← Singleton registry
│  - Loads YAML       │
│  - Matches models   │
│  - Creates config   │
└──────────┬──────────┘
           │
           ├─► ModelMatcher (pattern matching)
           │
           └─► ModelConfigurator (apply configs)
                    │
                    ├─► Environment variables
                    └─► GPU block overrides
```

## Adding a New Model

Simply add an entry to `model_configs.yaml`:

```yaml
models:
  your-org/your-model:
    architecture:
      model_type: llama  # Required: HF model type
      num_hidden_layers: 32  # Optional: for precise matching
      vocab_size: 128256  # Optional: for precise matching

    # Static batching configuration (pooling models only)
    static_batching_configs:
      - tp_size: 1
        warmup_shapes:
          - prompt_len: 512
            batch_size: 64

    # Continuous batching configuration (if supported)
    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 8192
        max_num_seqs: 16

      # With device config for TP=4
      - tp_size: 4
        max_model_len: 32768
        max_num_seqs: 32
        device_config:  # Optional: nested device configuration
          env_vars:
            VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
          num_gpu_blocks_override: 8192
```

**That's it!** No code changes needed for most models.

## Configuration Schema

### Architecture Pattern

Defines how to match a model from its HuggingFace config. Only `model_type` is
strictly required, but other fields are needed and used for more precise matching
(enough to distinguish between all other supported models).

```yaml
architecture:
  model_type: granite          # Required: HF model type
  num_hidden_layers: 40        # Optional: number of layers
  max_position_embeddings: 131072  # Optional: max positions
  hidden_size: 4096            # Optional: hidden dimension
  vocab_size: 49159            # Optional: vocabulary size
  num_key_value_heads: 8       # Optional: KV heads
  num_attention_heads: 32      # Optional: attention heads
  num_experts_per_tok: 0       # Optional: for MoE models
```

### Static Batching Configurations

For pooling models (embeddings, scoring) that use static batching:

```yaml
static_batching_configs:
  - tp_size: 1
    warmup_shapes:
      - prompt_len: 512
        batch_size: 64
      - prompt_len: 128
        batch_size: 8
```

**Note**: Decoder models (generation) only support continuous batching. Static batching is exclusively for pooling models.

### Continuous Batching Configurations

For models that support continuous batching. Each configuration can optionally
include a nested `device_config`:

```yaml
continuous_batching_configs:
  # Simple config without device overrides
  - tp_size: 1
    max_model_len: 3072
    max_num_seqs: 16

  # Config with device-specific settings
  - tp_size: 4
    max_model_len: 32768
    max_num_seqs: 32
    device_config:  # Optional, nested device configuration
      env_vars:
        VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
        FLEX_HDMA_P2PSIZE: 268435456
      num_gpu_blocks_override: 8192
```

### Device Configuration Templates (YAML Anchors)

To reduce verbosity, define reusable device configs as YAML anchors:

```yaml
device_config_templates:
  granite_8b_tp4_device_config: &granite_8b_tp4_device_config
    env_vars:
      VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
      FLEX_HDMA_P2PSIZE: 268435456
    num_gpu_blocks_override: 8192

models:
  ibm-granite/granite-3.3-8b-instruct:
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 32768
        max_num_seqs: 32
        device_config: *granite_8b_tp4_device_config  # Reference the anchor
```

## How It Works

### 1. Model Matching and Runtime Config Verification

Before SenDNN Inference loads the model, the registry:

1. Extracts the HuggingFace config
2. Compares it against all registered architecture patterns
3. Verifies there's a supported runtime configuration matching the runtime parameters:
   - For continuous batching: TP size, max_model_len, and max_num_seqs
   - For static batching: TP size AND warmup shapes must match
4. Returns a configurator with the appropriate device_config (if any)

```python
# In platform.py
from sendnn_inference.config.model_registry import get_model_registry


registry = get_model_registry()
# For static batching, pass warmup shapes for validation
warmup_shapes = cls._warmup_shapes if not envs_spyre.SENDNN_INFERENCE_USE_CB else None
configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)
```

### 2. Configuration Application

Once matched, the configurator applies environment variables and other
configurations as applicable. Overrides of the default values are allowed
as long as `SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG=0` (the default).

The configurator returns a summary of the applied configurations that can easily
be printed.

```python
if configurator:
    config_summary = configurator.configure(vllm_config)
    logger.info(config_summary.format_log_message())
```

### 3. Optional Features

All device configuration features are optional:

- **No device_config**: Model works with defaults
- **No env_vars**: No environment variables set
- **No num_gpu_blocks_override**: Uses vLLM's default calculation

## API Reference

### ModelConfigRegistry

```python
from sendnn_inference.config.model_registry import get_model_registry

registry = get_model_registry()

# Get configurator for runtime (recommended - does matching and verification)
# For static batching, pass warmup_shapes for validation
warmup_shapes = get_warmup_shapes()  # from platform
configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

# Find model by HF config (lower-level)
model_name = registry.find_matching_model(vllm_model_config)

# List all registered models
models = registry.list_models()
```

### ModelConfigurator

Default configurator that should be able to handle most models:

```python
class ModelConfigurator:
    def __init__(self, model_config: ModelConfig,
                 device_config: DeviceConfig | None = None):
        """Initialize with model config and optional device config"""

    def configure(self, vllm_config: VllmConfig) -> ConfigurationSummary:
        """Apply device configurations and return summary"""

    def set_env_var(self, key: str, value: Any, override: bool = False) -> ConfigValue:
        """Set environment variable with tracking"""

    def _configure_gpu_blocks(self, device_config: DeviceConfig,
                             vllm_config: VllmConfig) -> ConfigValue | None:
        """Configure GPU blocks with version-aware logic"""
```

### ConfigurationSummary

Returned by `configure()` to track what was applied:

```python
@dataclass
class ConfigurationSummary:
    model_name: str
    tp_size: int
    env_vars: dict[str, ConfigValue]  # Tracks default vs applied values
    num_blocks: ConfigValue | None    # GPU blocks override

    def format_log_message(self) -> str:
        """Format summary for logging with override warnings"""
```

### ConfigValue

Tracks configuration values with override detection:

```python
@dataclass
class ConfigValue:
    default: str | int | None  # Default value from config
    applied: str | int | None  # Applied value (possibly from user override)

    def was_overridden(self) -> bool:
        """Check if applied differs from default"""
```

## Examples

### Simple Embedding Model

```yaml
sentence-transformers/all-roberta-large-v1:
  architecture:
    model_type: roberta
    num_hidden_layers: 24
    vocab_size: 50265

  static_batching_configs:
    - tp_size: 1
      warmup_shapes:
        - prompt_len: 128
          batch_size: 8
```

### Complex Generation Model with Device Config

```yaml
# Define reusable device config template
device_config_templates:
  granite_8b_tp4_device_config: &granite_8b_tp4_device_config
    env_vars:
      VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
      FLEX_HDMA_P2PSIZE: 268435456
      FLEX_HDMA_COLLSIZE: 33554432
    num_gpu_blocks_override: 8192

models:
  ibm-granite/granite-3.3-8b-instruct:
    architecture:
      model_type: granite
      num_hidden_layers: 40
      max_position_embeddings: 131072
      hidden_size: 4096
      vocab_size: 49159
      num_key_value_heads: 8
      num_attention_heads: 32

    continuous_batching_configs:
      - tp_size: 1
        max_model_len: 3072
        max_num_seqs: 16
      - tp_size: 4
        max_model_len: 32768
        max_num_seqs: 32
        device_config: *granite_8b_tp4_device_config  # Reference anchor
```

## Troubleshooting

### Model Not Matched

If your model isn't being matched:

1. Check the HF config attributes: `print(model_config.hf_config.__dict__)`
2. Ensure `model_type` matches exactly
3. Add more specific attributes to narrow the match
4. Check logs with VLLM_LOGGING_LEVEL=DEBUG for matching attempts

### Configuration Not Applied

If configuration isn't being applied:

1. Verify the TP size matches a device_config key
2. Check that environment variables aren't already set
3. Review logs for configuration application messages
4. Ensure YAML syntax is correct

## Support

For questions or issues:

1. Check this README
2. Review existing model configurations in `model_configs.yaml`
3. Examine the configurator code in `configurators/model_configurator.py`
4. Consult the SenDNN Inference team
