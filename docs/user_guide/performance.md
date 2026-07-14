# Benchmarking and Performance

This page describes the recommended benchmarking methodology for SenDNN Inference and provides guidance for tuning performance. In addition to practical suggestions, it aims to give a solid background for understanding and interpreting results.

## Running Benchmarks

The recommended tool for performance benchmarking is the [vLLM bench CLI tool](https://docs.vllm.ai/en/stable/benchmarking/cli/), which is automatically included in the installation. In particular, we detail the use of [`vllm bench serve`](https://docs.vllm.ai/en/stable/benchmarking/cli/#online-benchmark), which evaluates end-to-end performance in a serving context and is well-suited for production-like deployment scenarios.

??? tip
    There are other vLLM benchmarking commands worth using as well:

    - [`vllm bench throughput`](https://docs.vllm.ai/en/stable/benchmarking/cli/#offline-throughput-benchmark) for measuring offline inference throughput
    - [`vllm bench sweep`](https://docs.vllm.ai/en/stable/benchmarking/sweeps/) which runs `vllm bench serve` across different parameter configurations, enabling comparison of different settings

    For multimodal models benchmarking, there are multiple approaches:

    - `vllm bench serve` with a custom multimodal (`custom_mm`) or a random multimodal (`random_mm`) dataset — more detailed information is given [below](#multimodal-considerations).
    - `vllm bench mm-processor` which specifically profiles the multimodal input processor pipeline in an offline fashion

How to use `vllm bench serve`:

1. Start the server:

```bash
vllm serve \
    --model {model} \
    --max-model-len {max-model-len} \
    --max-num-seqs {max-num-seqs}
```

See the [Supported Models](./supported_models.md) page for supported models and their respective configuration.

1. Run the benchmarking script (minimal command):

```bash
vllm bench serve \
    --backend vllm \
    --model {model} \
    --endpoint /v1/completions \
    --dataset-name {custom/sharegpt/random...} \
    --dataset-path {path to dataset (.json)} \
    --num-prompts {num-prompts} \
    --max-concurrency {num-concurrent-users} \
    --output-len {output-len}
```

!!! note

    When using the `custom` dataset, if the prompts already contain system instructions for your model, you probably want to use [`--skip-chat-template`](https://docs.vllm.ai/en/stable/cli/bench/serve/#-skip-chat-template) to avoid applying additional system instructions on top of the existing ones.

The following additional flags can help with insights and result interpretation:

- **`--metrics-percentiles 99,100`**: Sometimes recorded metrics are skewed enough that the P100 value differs significantly from P99, which distorts the mean. It is generally a good idea to collect the max value (P100).

- **`--percentile-metrics ttft,tpot,itl,e2el`**: By default, not all metrics are displayed. This flag ensures all metrics are reported. Metric descriptions are detailed [below](#metrics-description).

- The result visualization plots can be useful for debugging and interpretation:

    !!! info

        To generate these visualizations, you'll need to install the plotting libraries: `uv pip install vllm[bench]`

    - **Requests Statistics**: As explained below, the number of input and output tokens greatly affects performance. The **`--plot-dataset-stats`** flag saves a `.png` plot showing the number of input and output tokens for each request. See [vLLM Bench documentation — Dataset Statistics](https://docs.vllm.ai/en/stable/benchmarking/cli/#dataset-statistics).
    - **Interactive Timeline** ([vLLM Bench documentation — Interactive Timeline](https://docs.vllm.ai/en/stable/benchmarking/cli/#interactive-timeline)): the **`--plot-timeline`** flag along with **`--timeline-itl-thresholds {itl1},{itl2}`** generates an interactive `.html` file renderable in any modern web browser. This works best for short runs with relatively few requests.

- Save all results:
    - `--save-results`: saves results to a `.json` file in addition to the printed output
    - `--save-detailed`: saves individual recorded data per request (useful for debugging)
    - `--result-dir {path/to/results}`: target path for output results

### `--custom-output-len -1`

When running benchmarks, all requests typically use the same `max-tokens` value (the maximum number of output tokens for a request). This value can be set using [`--output-len`](https://docs.vllm.ai/en/stable/cli/bench/serve/#-output-len). For the `custom` dataset (`--dataset-name custom`), if the dataset contains per-request output token counts as shown in the [Custom dataset documentation](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/datasets/#vllm.benchmarks.datasets.CustomDataset), you can load the per-request `max-tokens` using `--custom-output-len -1`. Paired with `--ignore-eos` (which tells the model to ignore the EOS token and always generate exactly `max-tokens` tokens), this makes benchmarks more stable and reproducible, since the number of output tokens is fixed across runs. Without this, output length varies across runs — even at temperature 0.0, unless using [batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/#batch-invariance) — making results more variable and difficult to interpret.

!!! warning

    Using `--custom-output-len -1` paired with `--ignore-eos` has important implications:

    1. The `max-tokens` value is the *exact* number of tokens that will be generated. This means measured performance is optimistic and does not reflect what you would observe when setting a larger `max-tokens` value across all requests, because the scheduler doesn't need to account for a worst-case scenario. See the [Max-output-tokens](#max-output-tokens) section in performance tuning below.

    2. The number of output tokens in the dataset may differ significantly from what the model would naturally produce if allowed to stop at EOS. Depending on model verbosity, the performance may be very different. This method evaluates the inference stack in isolation, independent of model behavior.

### Multimodal Considerations

Benchmarking multimodal models adds additional configuration complexity. In addition to parameters such as text prompt length and generated output length that also apply to text-only benchmarking, request concurrency, number of multimodal items per request, image-size distribution, and input/output length should all be held constant across runs in order to provide a valid comparison. Changing these parameters changes the amount of multimodal preprocessing, vision encoding, KV-cache pressure, and scheduling pressure, so results are not directly comparable unless these values are kept consistent.

A typical online multimodal benchmark uses the OpenAI chat backend and the chat completions endpoint:

```bash
vllm bench serve \
    --backend openai-chat \
    --model {model} \
    --endpoint /v1/chat/completions \
    --dataset-name {custom-mm/random-mm...} \
    --dataset-path {path to dataset (.json)} \
    --random-input-len {input-len} \
    --random-output-len {output-len} \
    --num-prompts {num-prompts} \
    --max-concurrency {num-concurrent-users} \
    --random-mm-limit-mm-per-prompt '{"image": {num-images}, "video": 0}' \
    --random-mm-bucket-config '{(256, 256, 1): 0.5, (720, 1280, 1): 0.5}'  # this is the current default in vllm
```

The dataset used will depend on your use case. Prefer a `custom_mm` with a representative request dataset (provided with the `--dataset-path` parameter) for production-like evaluation and the `random-mm` dataset when the goal is to benchmark the inference stack under a controlled synthetic vision workload. See [documentation](https://docs.vllm.ai/en/stable/cli/bench/serve/#-dataset-name) for more details.

Some relevant flags for use with the `random-mm` dataset are:

- **`--random-mm-limit-mm-per-prompt`**: Sets per-request modality limits to control the number of images or videos attached to each request

- **`--random-mm-bucket-config`**: Maps `(height, width, num_frames)` buckets to sampling probabilities; a bucket with num_frames=1 represents an image

See the [vllm documentation](https://docs.vllm.ai/en/stable/benchmarking/cli/#synthetic-random-images-random-mm) for more details on these and other flags.

!!! note

    The image bucket distribution can have a large impact on TTFT and throughput. Larger images generally increase multimodal preprocessing and prefill cost, while a distribution with mixed image sizes can introduce more request-to-request variability.

## Metrics Description

A successful run outputs several metrics, described below in relation to concepts from the [scheduling and padding design document](../contributing/scheduler.md).

### TTFT

The time-to-first-token (TTFT) measures the time from when a request is sent by the client until the first token is received. It therefore includes time spent waiting in the queue and the full prefill time up to the last chunked prefill step. TTFT is heavily influenced by the [admission constraints](../contributing/scheduler.md#admission-constraints): if a request's last chunked prefill is held back for a long time — for example due to the volumetric or max-context-len constraint — the TTFT for that request, as well as the requests queued behind it, will be very high.

!!! warning

    The admission constraints consider the prompt length together with `max-output-tokens`. A high `max-output-tokens` value therefore has a large impact on TTFT and overall performance: requests are more likely to be blocked in the waiting queue, which directly increases observed TTFT values.

### ITL

The inter-token latency (ITL) is the time between two consecutive output tokens. Each request produces a list of ITL values — one per generated token after the first (the first token's latency is the TTFT). ITL reflects the decode time at each step, but since prefill interrupts decode (prefill and decode cannot run concurrently), a prefill step directly introduces ITL spikes. This is why prefill is chunked and interleaved with decode steps, to mitigate the intensity of the spikes.

!!! note
    Smaller chunk sizes improve ITL values: a smaller prefill chunk completes faster and causes a shorter decode interruption. On the other hand, a chunk size that is too small introduces overhead that hurts overall throughput.

### TPOT

The time-per-output-token (TPOT) is the mean time to generate the next decoded token for a given request — in other words, the mean of a request's ITL values. TPOT reflects average per-request performance, while ITL reveals individual latencies (revealing jittering and spikes).

### E2EL

The end-to-end latency (E2EL) is the time from submitting a request to the server until the last output token is received. This metric is not reported by default; use `--percentile-metrics ttft,tpot,itl,e2el` to enable it.

## Production metrics

vLLM exposes a number of metrics that can be used to monitor the health of the system. These metrics are exposed via the /metrics endpoint on the vLLM OpenAI compatible API server. Refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/usage/metrics/) for a list of the exposed metrics.

`sendnn-inference` adds a couple of metrics to vLLM, mainly related to request pausing:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `sendnn_inference:decode_batch` | Gauge | Number of requests in decode batch. |
| `sendnn_inference:paused` | Gauge | Number of paused requests. |
| `sendnn_inference:pause_events` | Counter | Count of times a request was paused. |
| `sendnn_inference:resume_events` | Counter | Count of times a request was resumed. |

## Visualization – Results Interpretation

For the run described in the [Scheduler Constraints visualization](../contributing/scheduler.md#visualization-scheduler-constraints), we show the corresponding timeline that would be obtained using `--plot-timeline`. The timeline is reconstructed for educational purposes (since request arrival steps cannot be precisely controlled with vllm bench serve), but it matches the expected shape. We assume a decode takes 1 unit of time and a chunked prefill takes 8 units.

**Observations**:

- The TTFT of request 0 corresponds to its prefill time (8 unit times) because it gets scheduled directly when joining (0 waiting time)

- The two first chunked prefill that request 1 undergoes during the decode of request 0 are reflected through two ITL spikes in request 0 (decode interrupted)

- If we didn't have prefill-decode interleaving, we would see fewer but longer ITL spikes. For example, we see five consecutive ITLS of 9 unit time in request 1. Those corresponds to the back-to-back prefills of request 2 and 3. Without interleaving, we would have one single long ITL of `5 * 8 + 1 (for the decode) = 41` unit times.

- As soon as one request gets stuck (due to various scheduling constraints), its TTFT increases, and so does the TTFT of all the requests queued behind it. This is the case for request 1 for example.

- As we can see with request 2, a long prompt with many tokens take multiple chunks to prefill and also delays the other requests.

<iframe src="../assets/plots/timeline_admission_constraints.html" width="100%" height="630px" frameborder="0"></iframe>

## Performance Tuning

The benchmark configuration has a large impact on observed performance. This section explains how to set parameters for realistic, high-quality results.

### Max-output-tokens

The [admission constraints](../contributing/scheduler.md#admission-constraints) use the total sequence length — prompt tokens plus requested output tokens — to decide whether to admit a request. A high `max-output-tokens` value increases the total sequence length, making requests wait longer before being admitted. Keeping this value as low as possible is therefore important for minimizing TTFT and improving overall throughput.

### Max-concurrency

The `--max-concurrency` argument sets the maximum number of active requests that the client will maintain. For a value `n`, the client starts by sending `n` requests, then sends a new request each time one completes, keeping exactly `n` in flight at all times. In short, `--max-concurrency` represents the number of concurrent users interacting with the server.

??? tip

    Instead of `--max-concurrency`, you can control the request arrival rate using [`--request-rate`](https://docs.vllm.ai/en/stable/cli/bench/serve/#-request-rate) paired with [`--burstiness`](https://docs.vllm.ai/en/stable/cli/bench/serve/#-burstiness):

    - `--request-rate`: number of requests per second sent to the server. Defaults to infinity (all requests are sent at the start).
    - `--burstiness`: a factor ranging from 0.0 (extreme bursts) to infinity (perfectly constant rate), used as the parameter of a Gamma distribution over inter-request arrival times.

The [volumetric constraint](../contributing/scheduler.md#last-chunk-constraints) prevents too many long requests from running simultaneously. Depending on the sequence length distribution of your dataset, even when compiling for a large batch size, the actual number of concurrently running requests may be small because of this constraint. We therefore recommend setting `--max-concurrency` to approximately `max-volume-limit / top-percentile-sequence-length`. Increasing concurrency well beyond this point yields diminishing throughput gains while causing TTFT to increase sharply.

!!! example

    With the current max volume limit of 131,072 and sequence lengths (prompt tokens + max output tokens) up to 32,768 tokens, at most 4 requests can run when there is one long request in the batch: `4 × 32,768 = 131,072`. In this case, set `--max-concurrency` to 4, even if the model is compiled for a larger batch size.

### Num Prompts

If [`--num-prompts`](https://docs.vllm.ai/en/stable/cli/bench/serve/?query=custom-output-len#-num-prompts) is set to a value higher than the number of requests in the dataset, requests will be upsampled (i.e., duplicated). Due to prefix caching, duplicate requests will get a higher cache hit frequency than they would without upsampling, artificially inflating performance results.

### Requests Ordering

By default, requests are sent to the server in shuffled order. Sending them in the original order can be enabled with [`--disable-shuffle`](https://docs.vllm.ai/en/stable/cli/bench/serve/?query=custom-output-len#-disable-shuffle).

The effect of ordering on performance is highly workload-dependent. For datasets where consecutive requests share a common prefix — such as multi-turn chat — shuffling can cause a shared prefix to be evicted from the cache before the next request that needs it arrives, artificially reducing cache hit rate and degrading performance. In contrast, keeping the original order can significantly increase cache utilization.
