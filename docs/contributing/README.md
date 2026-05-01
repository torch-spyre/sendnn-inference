# Contributing to SenDNN Inference

Thank you for your interest in contributing to the Spyre plugin for vLLM! There are several ways you can contribute:

- Identify and report any issues or bugs.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

## Issues

If you encounter a bug or have a feature request, please search [existing issues](https://github.com/torch-spyre/sendnn-inference/issues?q=is%3Aissue) first to see if it has already been reported. If not, please create a new issue, by using our [issue templates](https://github.com/torch-spyre/sendnn-inference/issues/new/choose):

- **🐛 Bug Report**: For reporting bugs and unexpected behavior
- **🚀 Feature Request**: For suggesting new features or improvements

You can also reach out for support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace.

## Docs

### Building the docs with MkDocs

#### Install MkDocs and Plugins

Install MkDocs along with the [plugins](https://github.com/torch-spyre/sendnn-inference/blob/main/mkdocs.yaml) used in the SenDNN Inference documentation.

```bash
uv pip install -r docs/requirements-docs.txt
```

!!! note
    Ensure that your Python version is compatible with the plugins (e.g., `mkdocs-awesome-nav` requires Python 3.10+)

#### Start the Development Server

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it.

Make sure you're in the same directory as the `mkdocs.yml` configuration file in the `sendnn-inference` repository, and then start the server by running the `mkdocs serve` command:

```bash
mkdocs serve
```

Example output:

```console
INFO    -  Documentation built in 106.83 seconds
INFO    -  [22:02:02] Watching paths for changes: 'docs', 'mkdocs.yaml'
INFO    -  [22:02:02] Serving on http://127.0.0.1:8000/
```

#### View in Your Browser

Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to see a live preview.

#### Learn More

For additional features and advanced configurations, refer to the official [MkDocs Documentation](https://www.mkdocs.org/).

## Testing

!!! tip
    When running tests, if errors occur, these can be analyzed/debugged by setting `DISABLE_ASSERTS = True` in spyre_util.py and by rerunning the test using `pytest --capture=no tests/spyre/test_spyre_basic.py`. After debugging, `DISABLE_ASSERTS` should be reset to `False`.

### Testing Locally on CPU (No Spyre card)

Optionally, download the `ibm-ai-platform/micro-g3.3-8b-instruct-1b` model:

```sh
python -c "from transformers import pipeline; pipeline('text-generation', model='ibm-ai-platform/micro-g3.3-8b-instruct-1b')"
```

!!! caution
    The Hugging Face API download does **not** work on `arm64`.

By default, the model is saved to `.cache/huggingface/hub/models--ibm-ai-platform--micro-g3.3-8b-instruct-1b`.

Then, source the environment variables:

```sh
source _local_envs_for_test.sh
```

Optionally, install development dependencies:

```sh
uv pip install --group dev
```

Now, you can run the tests:
  
```sh
python -m pytest -v -x tests -m "cpu and e2e"
```

Here is a list of `pytest` markers you can use to filter them:

```python
--8<-- "pyproject.toml:test-markers-definition"
```

### Testing Specific Features

For most of the supported features the testing code can be run in isolation by passing
the appropriate marker to pytest. Note the markers can be combined with boolean logic
operators "and", "or" "not" and parentheses "()".

- **prefix_caching**: Runs only the prefix caching tests
- **quantized**: Runs all the tests with quantized models weights (FP8)
- **embedding**: Runs only embedding model tests
- **scoring**: Runs only reranker or scoring model tests
- **multimodal**: Runs only multimodal model tests

Example, run the prefix caching tests:

```sh
python -m pytest -v -x tests/e2e -m prefix_caching
```

## Debugging

!!! tip
    You can `oc edit` a pod and change the image without having the pod schedule to a different node. This can be useful for testing whether software or hardware is the issue.

- The script `/opt/sentient/bin/aiu-query-devices` in the pod can be used to see the connectivity between the `AIUs` on the machine. You can also infer this from environment variables with names like `AIU_TIER_\d_SET_\d_RANK_\d`.
  
- `SPYRE_DEVICES` can be used to select which devices will be selected for each `RANK`. This is similar to how `CUDA_VISIBLE_DEVICES` works for GPU.
  
    !!! example
        `0,2,4,6` will assign rank `0` to AIU index `0`, rank `1` to AIU index `2`, rank `2` to AIU index `4`, and rank `3` to AIU index `6`.
  
    - An alternative is to use `AIU_WORLD_RANK_\d=0000:aa:00.0` to explicitly map ranks to `PCI` addresses (make sure there are no duplicates used at runtime).
  
- A bash script that uses `/opt/sentient/senlib/bin/senlib_unit_test` to check each `AIU` allocated to the pod to see if they work for a basic test:
  
    ```shell
    --8<-- "tools/check_aiu.sh"
    ```

### Logging levels

Various log levels that can be configured:

- `DTLOG_LEVEL` - `TRACE, DEBUG, INFO, WARNING, ERROR`
- `TORCH_SENDNN_LOG` - `WARNING, CRITICAL`
- `VLLM_LOGGING_LEVEL` - `DEBUG, INFO, WARNING, ERROR`
- `DT_DEEPRT_VERBOSE` - `0, -1`

!!! tip
    `DTLOG_LEVEL=INFO` (piped to file) can help you see what device addresses are actually in use. Look for the string `Opened: SEN:VFIO`.

!!! tip

    Set `DT_DEEPRT_VERBOSE` to 0 to enable verbose compiler prints for debugging.

!!! tip
    In order to stop massive log spew, this configuration is ideal:
    ```
    export DTLOG_LEVEL=ERROR
    export TORCH_SENDNN_LOG=CRITICAL
    ```

For tensor-parallel debugging, you can enable an option to redirect all log output from each rank to an individual file.
Set `SENDNN_INFERENCE_WORKER_LOG_REDIRECT_DIR` to a local directory, and each rank will redirect stdout and stderr into their own file inside the directory.
This can be helpful to avoid having interleaved stack dumps from different ranks in stderr.

### Performance Metrics

When deploying to kubernetes clusters, prometheus + grafana can be installed and configured to scrape metrics from vLLM's `/metrics` endpoint.

vLLM can also be configured to log performance metrics about every request to a local file.
Setting both `SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED=1` and `SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR=/some/path` and ensuring that vLLM stat logging is enabled will generate metrics in `/some/path/request_metrics.jsonl`. A sample of this file looks like:

```json
{"timestamp": "2025-10-10T12:25:17.544", "prefill_interrupt_seconds": 0, "decode_only_itl_seconds": 0.05045744727055232, "finish_reason": 1, "num_prompt_tokens": 1, "num_generation_tokens": 16, "max_tokens_param": 16, "e2e_latency_seconds": 0.9784879684448242, "queued_time_seconds": 6.0582999140024185e-05, "prefill_time_seconds": 0.220398832927458, "inference_time_seconds": 0.9772605419857427, "decode_time_seconds": 0.7568617090582848, "mean_time_per_output_token_seconds": 0.05045744727055232}
{"timestamp": "2025-10-10T12:25:19.632", "prefill_interrupt_seconds": 0, "decode_only_itl_seconds": 0.10008190000274529, "finish_reason": 1, "num_prompt_tokens": 1, "num_generation_tokens": 16, "max_tokens_param": 16, "e2e_latency_seconds": 2.0864057540893555, "queued_time_seconds": 0.2935298749944195, "prefill_time_seconds": 0.1466117500094697, "inference_time_seconds": 1.647840250050649, "decode_time_seconds": 1.5012285000411794, "mean_time_per_output_token_seconds": 0.10008190000274529}
{"timestamp": "2025-10-10T12:25:19.632", "prefill_interrupt_seconds": 0.14661192893981934, "decode_only_itl_seconds": 0.1000875825372835, "finish_reason": 1, "num_prompt_tokens": 1, "num_generation_tokens": 16, "max_tokens_param": 16, "e2e_latency_seconds": 2.0864808559417725, "queued_time_seconds": 0.1469848749693483, "prefill_time_seconds": 0.14646116609219462, "inference_time_seconds": 1.7943868330912665, "decode_time_seconds": 1.6479256669990718, "mean_time_per_output_token_seconds": 0.10986171113327145}
{"timestamp": "2025-10-10T12:25:19.632", "prefill_interrupt_seconds": 0.29317212104797363, "decode_only_itl_seconds": 0.10008799746477355, "finish_reason": 1, "num_prompt_tokens": 1, "num_generation_tokens": 16, "max_tokens_param": 16, "e2e_latency_seconds": 2.08658504486084, "queued_time_seconds": 0.0001724999165162444, "prefill_time_seconds": 0.14670966705307364, "inference_time_seconds": 1.9412017500726506, "decode_time_seconds": 1.794492083019577, "mean_time_per_output_token_seconds": 0.11963280553463847}
{"timestamp": "2025-10-10T12:25:19.632", "prefill_interrupt_seconds": 0.4400491714477539, "decode_only_itl_seconds": 0.10009045804229875, "finish_reason": 1, "num_prompt_tokens": 1, "num_generation_tokens": 16, "max_tokens_param": 16, "e2e_latency_seconds": 2.0868380069732666, "queued_time_seconds": 2.9250048100948334e-05, "prefill_time_seconds": 0.1447284579044208, "inference_time_seconds": 2.086134499986656, "decode_time_seconds": 1.9414060420822352, "mean_time_per_output_token_seconds": 0.12942706947214902}
```

### Topology Aware Allocation

This section is specific to the AIU operator and scheduling workloads onto specific cards.

(TODO: link to docs once they exist)

- This mode supports users to request a special set of AIU cards based on `PCI` topology. By using this mode, we can guarantee to pick up AIU cards of a particular class in the node:
  
    - `Tier0` provides a set of cards in the same `PCI` switch.
    - `Tier1` provides a set of cards from at most one-hop away `PCI` switch.
    - `Tier2` provides a set of cards from at most two-hops away `PCI` switch.

- Running a Multi AIU Job using `ibm.com/aiu_pf_tier0,tier1,tier2`:
  
    - This resource type is used for picking up a topology aware card set, which is required to run tensor parallel (`TP`) workloads more effectively. By using `tierX` class resource, `TP` users can automatically get a best performing card set for the workload.

- The maximum number of allocatable resources in each tier depends on the platform & cluster, but we can get up to:
  
    - `Tier0` - `4` cards
    - `Tier1` - `8` cards
    - `Tier2` - `16` cards

- Devices in `tier0` can do `peer-to-peer (P2P) RDMA`, devices on different trees use `Host DMA` sharing files through `/dev/shm`.

    !!! warning
         If you request cards greater than the cards supported by the switch, the pod will never be scheduled. In the above example, if you specify `ibm.com/aiu_pf_tier0: 5` in your yaml, the pod will never be scheduled because the maximum set of cards in `tier0` was specified as `4`.

## Pull Requests

### Linting

When submitting a PR, please make sure your code passes all linting checks. We use prek with a .pre-commit-config.yaml file to run checks on every commit.

The `format.sh` script will run prek from an isolated virtual environment using [uvx](https://docs.astral.sh/uv/guides/tools/). The only requirement is that you have `uv` installed.

```sh
bash format.sh
```

Alternatively, you can [install prek](https://github.com/j178/prek?tab=readme-ov-file#installation) and set up a git hook to run it on every commit with:

```sh
prek install
```

### DCO and Signed-off-by

When contributing, you must agree to the [DCO](https://github.com/torch-spyre/sendnn-inference/blob/main/DCO). Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## License

See <gh-file:LICENSE>.
