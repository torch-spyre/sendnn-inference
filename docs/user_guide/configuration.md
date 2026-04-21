# Configuration

For a complete list of configuration options, see [Environment Variables](env_vars.md).

## SenDNN Inference 2.0 Migration Guide

This guide covers breaking changes introduced in SenDNN Inference 2.0, including the package rename from `vllm_spyre` to `sendnn_inference` and associated configuration changes.

### Package Rename

The Python package has been renamed from `vllm_spyre` to `sendnn_inference`. This affects:

- **Installs from source** Update your remote to [https://github.com/torch-spyre/sendnn-inference](https://github.com/torch-spyre/sendnn-inference)
- **Installs from PyPI**: Install with `pip install sendnn-inference` (was `vllm-spyre`)

### VLLM_PLUGINS Configuration

**Breaking Change**: The vLLM plugin entry point has changed from `spyre` to `sendnn_inference`.

Update your environment configuration:

```shell
# Old (vllm-spyre 1.x)
export VLLM_PLUGINS=spyre

# New (sendnn-inference 2.0)
export VLLM_PLUGINS=sendnn_inference
```

This change is required for vLLM to discover and load the SenDNN Inference platform plugin. Without this update, vLLM will fail to initialize the Spyre backend.

### Environment Variable Renaming

**Breaking Change**: All `VLLM_SPYRE_*` environment variables have been renamed to `SENDNN_INFERENCE_*`.

**Example migration**:

```shell
# Old (vllm-spyre 1.x)
export VLLM_SPYRE_DYNAMO_BACKEND=sendnn
export VLLM_SPYRE_WARMUP_PROMPT_LENS=64,128
export VLLM_SPYRE_WARMUP_BATCH_SIZES=1,4

# New (sendnn-inference 2.0)
export SENDNN_INFERENCE_DYNAMO_BACKEND=sendnn
export SENDNN_INFERENCE_WARMUP_PROMPT_LENS=64,128
export SENDNN_INFERENCE_WARMUP_BATCH_SIZES=1,4
```

!!! note
    The old `VLLM_SPYRE_*` variable names are no longer recognized. All configuration must use the new `SENDNN_INFERENCE_*` prefix.

For a complete list of environment variables, see [Environment Variables](env_vars.md).

### Runtime Behavior Changes

For users migrating from SenDNN Inference 1.x releases, the default configuration for generative models has changed.
All models now run with chunked prefill, and prefix caching is enabled by default.

The following environment variables can be removed from all configurations:

```shell
SENDNN_INFERENCE_USE_CB
SENDNN_INFERENCE_USE_CHUNKED_PREFILL
```

Configuring the static chunk size for chunked prefill is now only done via `--max-num-batched-tokens`.
If your deployment had an override set with `VLLM_DT_CHUNK_LEN`, use the CLI argument instead.
The default value for the chunk size is 512.

Because prefix caching is enabled by default, the `--enable-prefix-caching` CLI arg can be removed from all deployments.
To disable prefix caching, use `--no-enable-prefix-caching`.

Support for static batching with generative models has been removed.
Any deployments for *generative* (not pooling) models with static batch shapes will now run in chunked prefill mode instead.
If you have the following environment variable set, you likely need to re-evaluate your deployment:

```shell
SENDNN_INFERENCE_WARMUP_NEW_TOKENS
```

There are no breaking changes to deployments for pooling models with SenDNN Inference 2.x.

## Backend Selection

The torch.compile backend can be configured with the `SENDNN_INFERENCE_DYNAMO_BACKEND` environment variable.

All models can be tested on CPU by setting this to `eager`.
To run inference on IBM Spyre Accelerators, this should be set to `sendnn`.

For systems with no IBM Spyre Accelerators, this can be set to `sendnn_compile_only` to use the same stack as the `sendnn` backend on CPU in order to create compilation artifacts usable by IBM Spyre Accelerators with the `sendnn` backend.

Support for the vLLM v0 backend has been removed, only the vLLM v1 backend is supported.

## Generative Models

When running decoder models for text generation, SenDNN Inference uses dynamic batching with chunked prefill and automatic prefix caching.
This looks and feels like running vllm on any other accelerator, with a few minor differences.

### Chunked Prefill

Chunked prefill is a technique that improves Inter-Token Latency (ITL) in continuous batching mode when large prompts need to be prefetched. Without it, these large prefills can negatively impact the performance of ongoing decodes. In essence, chunked prefill divides incoming prompts into smaller segments and processes them incrementally, allowing the system to balance prefill work with active decoding tasks.

For configuration and tuning guidance, see the [vLLM official documentation on chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill).

As in vLLM, the `max_num_batched_tokens` parameter controls how chunks are formed. While vLLM can dynamically schedule mixed batches of prefill and decode with arbitrary chunk sizes, the SenDNN Inference implementation is limited to compiling prefill programs for a single fixed chunk size.
SenDNN Inference interleaves decode passes with these fixed-chunk-size prefill passes to emulate chunked prefill. The `max_num_batched_tokens` parameter controls this fixed chunk size for prefill passes in SenDNN Inference.

This parameter should be tuned according to your infrastructure, it is recommended to set it from `512` to `4096` tokens and it **must** be multiple of the block size (currently fixed to `64`). For convenience, when using the model `ibm-granite/granite-3.3-8b-instruct` with `tp=4`, SenDNN Inference automatically sets `max_num_batched_tokens` to `512`, a value known to produce good hardware utilization in this setup.

In chunked prefill mode, the `vllm:kv_cache_usage_perc` metric will report the correct KV cache usage on the Spyre cards for all active requests.

### Prefix Caching

When running generative models, prefix caching is enabled by default, and can be disabled with the  `--no-enable-prefix-caching` CLI flag. An overview of prefix caching can be found in the [vLLM official documentation on Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/#limits).

Prefix caching mirrors upstream vLLM, though the requirement for fixed-size prefill chunks means the number of chunks in a prefill is only reduced if an entire chunk is available in cache. Therefore, workloads may show lower hit rates when compared to other accelerators.

When prefix caching is enabled, the `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` metrics correctly report prefix cache stats in tokens.

## Pooling Models

For the embedding, scoring, and reranking tasks, vLLM supports running Pooling Models. More information on Pooling Models can be found in the [vLLM official documentation](https://docs.vllm.ai/en/latest/models/pooling_models/).

SenDNN Inference runs all pooling models using static batching, where graphs are pre-compiled for each configured batch shape. This adds extra constraints on the sizes of inputs for each request, and requests that do not fit the precompiled graphs will be rejected.

!!! caution
    There are no up-front checks that the compiled graphs will fit into the available memory on the Spyre cards. If the graphs are too large for the available memory, vllm will crash during model warmup.

These batch shapes must be configured with the `SENDNN_INFERENCE_WARMUP_*` environment variables. For example, to warm up two graph shapes for one single large request and four smaller requests you could use:

```shell
export SENDNN_INFERENCE_WARMUP_BATCH_SIZES=1,4
export SENDNN_INFERENCE_WARMUP_PROMPT_LENS=4096,1024
```

!!! note
    The standard CLI args `--max-num-seqs` and `--max-model-len` are ignored for all pooling models, and prefix caching is not supported.

## Caching Compiled Graphs

`torch_sendnn` supports caching compiled model graphs, which can vastly speed up warmup time when loading models in a distributed setting.

To enable this, set `TORCH_SENDNN_CACHE_ENABLE=1` and configure `TORCH_SENDNN_CACHE_DIR` to a directory to hold the cache files. By default, this feature is disabled.

### Require Precompiled Decoders

Model compilation can be resource intensive and disruptive in production environments. To mitigate this, artifacts stored in `TORCH_SENDNN_CACHE_DIR` can be persisted to a shared volume during pre-deployment. Requiring the server to load from the cache avoids unexpected recompilation on the inference server.

To enforce the use of precompiled models, set:

```sh
SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS=1
```

and create an empty catalog file:

```sh
echo '{}' > ${TORCH_SENDNN_CACHE_DIR}/pre_compiled_cache_catalog.json
```

This configuration ensures that if a precompiled model is not found, an error will be raised. The catalog file is mandatory and serves as metadata for precompiled models in the cache. It enables the server to surface useful information and warnings in logs tagged with `[PRECOMPILED_WARN]`.

Catalog checks inspect metadata of the launch configuration that affect the cached artifacts including:

- vLLM configurations (tensor parallelism, batch size, static vs. continuous batching)
- Library versions used during precompilation
- Model name

If a matching entry is not found in the catalog, the server will still attempt to load from the cache. This allows precompiled models without catalog metadata to be used. However, if no precompiled model exists in the cache, the system will raise:

```text
RuntimeError: Compilation disabled
```

Scripts to generate and update pre_compiled_cache_catalog.json will be provided in future releases.

!!! note
    This feature is only available for generative models, pooling models are not supported.
