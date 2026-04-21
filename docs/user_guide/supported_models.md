# Supported Models

The SenDNN Inference plugin relies on model code implemented by the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack/tree/main/fms/models).

## Verified Deployment Configurations

The following models have been verified to run on SenDNN Inference with the listed
configurations. These tables are automatically generated from the model configuration file.

### Generative Models

Models with continuous batching support for text generation tasks.

<!-- GENERATED_GENERATIVE_MODELS_START -->
<!-- GENERATED_GENERATIVE_MODELS_END -->

### Pooling Models

Models with static batching support for embedding and scoring tasks.

<!-- GENERATED_POOLING_MODELS_START -->
<!-- GENERATED_POOLING_MODELS_END -->

## Model Configuration

The Spyre engine uses a model registry to manage model-specific configurations. Model configurations
are defined in <gh-file:sendnn_inference/config/model_configs.yaml> and include:

- Architecture patterns for model matching
- Device-specific configurations (environment variables, GPU block overrides)
- Supported runtime configurations (static batching warmup shapes, continuous batching parameters)

When a model is loaded, the registry automatically matches it to the appropriate configuration and
applies model-specific settings.

### Configuration Validation

By default, the Spyre engine will log warnings if a requested model or configuration is not found
in the registry. To enforce strict validation and fail if an unknown configuration is requested,
set the environment variable:

```bash
export SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG=1
```

When this flag is enabled, the engine will raise a `RuntimeError` if:

- The model cannot be matched to a known configuration
- The requested runtime parameters are not in the supported configurations list

See the [Configuration Guide](configuration.md) for more details on model configuration.
