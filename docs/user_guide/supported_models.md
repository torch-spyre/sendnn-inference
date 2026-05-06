# Supported Models

The SenDNN Inference plugin relies on model code implemented by the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack/tree/main/fms/models).

## Verified Deployment Configurations

The following models have been verified to run on SenDNN Inference with the listed
configurations. These tables are automatically generated from the model configuration file.

### Generative Models

Models with continuous batching support for text generation tasks.

<!-- GENERATED_GENERATIVE_MODELS_START -->


**[ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)**

| Max Model Len | Max Num Seqs | Tensor Parallel Size |
|---------------|--------------|----------------------|
| 3072 | 16 | 1 |
| 8192 | 4 | 1 |
| 8192 | 4 | 2 |
| 32768 | 32 | 4 |


**[ibm-granite/granite-3.3-8b-instruct-FP8](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct-FP8)**

| Max Model Len | Max Num Seqs | Tensor Parallel Size |
|---------------|--------------|----------------------|
| 3072 | 16 | 1 |
| 16384 | 4 | 4 |
| 32768 | 32 | 4 |


**[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)**

| Max Model Len | Max Num Seqs | Tensor Parallel Size |
|---------------|--------------|----------------------|
| 3072 | 16 | 1 |
| 16384 | 4 | 4 |
| 32768 | 32 | 4 |


**[mistralai/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)**

| Max Model Len | Max Num Seqs | Tensor Parallel Size |
|---------------|--------------|----------------------|
| 8192 | 32 | 2 |
| 32768 | 32 | 4 |


**[mistralai/Ministral-3-14B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16)**

| Max Model Len | Max Num Seqs | Tensor Parallel Size |
|---------------|--------------|----------------------|
| 32768 | 32 | 4 |

<!-- GENERATED_GENERATIVE_MODELS_END -->

### Pooling Models

Models with static batching support for embedding and scoring tasks.

<!-- GENERATED_POOLING_MODELS_START -->


**[ibm-granite/granite-embedding-125m-english](https://huggingface.co/ibm-granite/granite-embedding-125m-english)**

| SENDNN_INFERENCE_WARMUP_BATCH_SIZES | SENDNN_INFERENCE_WARMUP_PROMPT_LENS | Tensor Parallel Size |
|-------------------------------------|-------------------------------------|----------------------|
| 64 | 512 | 1 |


**[ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)**

| SENDNN_INFERENCE_WARMUP_BATCH_SIZES | SENDNN_INFERENCE_WARMUP_PROMPT_LENS | Tensor Parallel Size |
|-------------------------------------|-------------------------------------|----------------------|
| 64 | 512 | 1 |


**[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)**

| SENDNN_INFERENCE_WARMUP_BATCH_SIZES | SENDNN_INFERENCE_WARMUP_PROMPT_LENS | Tensor Parallel Size |
|-------------------------------------|-------------------------------------|----------------------|
| 64 | 512 | 1 |


**[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)**

| SENDNN_INFERENCE_WARMUP_BATCH_SIZES | SENDNN_INFERENCE_WARMUP_PROMPT_LENS | Tensor Parallel Size |
|-------------------------------------|-------------------------------------|----------------------|
| 1 | 8192 | 1 |


**[BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)**

| SENDNN_INFERENCE_WARMUP_BATCH_SIZES | SENDNN_INFERENCE_WARMUP_PROMPT_LENS | Tensor Parallel Size |
|-------------------------------------|-------------------------------------|----------------------|
| 64 | 512 | 1 |


**[sentence-transformers/all-roberta-large-v1](https://huggingface.co/sentence-transformers/all-roberta-large-v1)**

| SENDNN_INFERENCE_WARMUP_BATCH_SIZES | SENDNN_INFERENCE_WARMUP_PROMPT_LENS | Tensor Parallel Size |
|-------------------------------------|-------------------------------------|----------------------|
| 8 | 128 | 1 |

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
