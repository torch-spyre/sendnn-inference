# Maintaining SenDNN Inference

This page describes a few key maintenance tasks and how to accomplish them.

## vLLM Compatibility

When a new version of vLLM is released, the plugin usually needs to be updated to both:

- Work with any newly released vLLM changes
- Continue working with the previous versions of vLLM

### Updating vLLM version

To update the version of vLLM used by the plugin, the pyproject.toml needs to be updated in two places:

1. [tool.uv.sources.vllm](https://github.com/torch-spyre/sendnn-inference/blob/v2.0.0-rc.1/pyproject.toml#L79)
2. [project.dependencies](https://github.com/torch-spyre/sendnn-inference/blob/v2.0.0-rc.1/pyproject.toml#L16)

!!! note
    We specify the source tag for vLLM here because we generally need to install vLLM from source to avoid
    pulling in the precompiled cuda kernels and cuda dependencies.

Once the pyproject.toml is updated, the lockfile needs to be updated as well. The simplest way to do
that is to run `uv sync`. More information about uv locking can be found [in the uv docs](https://docs.astral.sh/uv/concepts/projects/sync/).

Now that the pyproject.toml and uv.lock files are upgraded, the default set of test suites will run
with the new version of vLLM, and you can move on to fixing the failures.

### Adding forward compatibility

When updating sendnn-inference code for a new version of vLLM, it's important to ensure that the code remains
backwards compatible with older releases. This usually means conditionally adding new method arguments or
new dataclass fields. We provide two utilities in [compat_utils](https://github.com/torch-spyre/sendnn-inference/blob/v2.0.0-rc.1/sendnn_inference/compat_utils.py)
to make this easy to do.

For example:

```python
# Only pass the `c` argument if the `foo` function accepts it
kwargs = {"c": some_value} if has_argument(foo, "c") else {}

foo(a, b, **kwargs)
```

Whenever code like this is added, we always make sure to add a test that will begin failing once the
lowest supported vLLM version no longer requires the backwards compatibility. This allows us to clean
up the code on the go as we drop older vLLM support. See [examples of compatibility tests here](https://github.com/torch-spyre/sendnn-inference/blob/v1.8.0/tests/utils/test_upstream_compatibility.py).

### Testing all supported vLLM versions

We maintain a matrix of test jobs that runs a small set of tests on every supported version of vLLM.
This helps to ensure that our backwards compatibility is implemented correctly.

When adding a new version of vLLM, we need to add an entry to this test matrix to add a test job for
the version of vLLM that we just upgraded from. See examples of [the test matrix here](https://github.com/torch-spyre/sendnn-inference/blob/v1.8.0/.github/workflows/test.yml#L83-L173).

### Removing support for a vLLM version

We periodically raise the vLLM lower bound once all key stakeholders agree it is no longer needed.

To do this:

1. Update the lower bound in [project.dependencies](https://github.com/torch-spyre/sendnn-inference/blob/v2.0.0-rc.1/pyproject.toml#L16)
2. Update the [test matrix](https://github.com/torch-spyre/sendnn-inference/blob/v1.8.0/.github/workflows/test.yml#L83-L173) to remove the tests for any versions of vllm that we removed support for
    1. Ensure that the new lowest vLLM version is run with `vllm_version.name: "vLLM:lowest"` and runs the `compat` marker
3. For any failing `compat` tests, remove the backwards compatibility code that is no longer required

## Managing Dependencies

Dependencies for sendnn-inference are managed with the `uv` tool. Usually, updating a dependency is as easy as running `uv add`, which will update both the pyproject.toml and the uv.lock file. An example is:

```shell
uv add "ibm-fms>=1.8.0,<2.0"
```

To update the locked version of a dependency without changing the version bounds in pyproject.toml, use `uv sync --upgrade-package`. An example is:

```shell
uv sync --upgrade-package ibm-fms
```

This would be done when you don't need to change the lower bound for a given dependency, but do want to lock a newer version to install in builds.
For a general refresh of all dependencies, use `uv sync --upgrade`.

By default, `uv` requires that the entire set of dependencies can successfully resolve and that there are no conflicts. This is sometimes problematic, as packages will often not declare forwards compatibility with future versions of their own dependencies, requiring consumers to wait for new releases. For example, maybe a new version of ibm-fms requires `transformers>=4.57`, but the latest vllm release still requires `transformers<4.57`.
To work around these conflicts, we use the `tool.uv.override-dependencies` section of our pyproject.toml file. In this case, we could specify `transformers==4.57.0` to ignore vllm's upper bound on transformers. Doing so comes with the obvious risks that there may be incompatibilities, and we should try to keep the overrides to a minimum.

## New Model Support

### Generative Models

All decoder models used for text generation must be implemented in [FMS](https://github.com/foundation-model-stack/foundation-model-stack) first.
FMS takes care of ensuring that the model code is compatible with the spyre hardware and the torch_sendnn
compilation pathway.

In order to be ready to run on sendnn-inference, the models must pass tests with [AFTU](https://github.com/foundation-model-stack/aiu-fms-testing-utils) with paged attention and chunked prefill enabled.

Once the model is ready to run on spyre hardware, it can be added to sendnn-inference by:

1. Updating the [model_loader](https://github.com/torch-spyre/sendnn-inference/blob/v2.0.0-rc.1/sendnn_inference/model_executor/model_loader/spyre.py#L109-L130) to allow the model architecture, making any necessary adjustments
2. Adding working configuration overrides to [model_configs.yaml](https://github.com/torch-spyre/sendnn-inference/blob/v2.0.0-rc.1/sendnn_inference/config/model_configs.yaml)

For multimodal models, see the docs on [adding multimodal models](./multimodal/adding_new_models.md)

!!! note
    If model configurations are not applied properly, a common failure mode is that the device will run out of memory with `DtException: Need to find a valid memory space`.
    This is because we typically need to limit the KV cache size to be smaller than `max_num_seqs * max_model_len`, and accordingly we limit the maximum dynamic batch area with `VLLM_DT_MAX_BATCH_TKV_LIMIT` to avoid either running out of KV cache space or attempting to run a batch shape that the compiler did not compile for.

### Pooling Models

Pooling models all use hf transformers code, and compile for static batch shapes.
Simply try running one, and if it works, record the working configuration in the model configs YAML file.

We don't have any guidance available for enabling pooling models if they do not compile successfully.

## Testing changes on Spyre hardware

To run tests on Spyre hardware, use the webhook integration with the Spyre validation suite. This is triggered by PR comments starting with `bot:test`.

!!! note
   System access and allow-listing are required to trigger these runs.

You can control the test scope by specifying Pytest markers and files within the PR comment. These are specified with `MARKERS` and `TEST_FILE`, for example:

```markdown
bot:test
MARKERS="spyre and prefix_caching"
TEST_FILE=tests/e2e/test_spyre_basic.py
```

The `spyre-ci` status will be posted back to the PR once the job completes, but we do not strictly require this passing check to merge PRs.

For any changes to dependencies or changes to model runner code, a thorough set of tests with the `spyre` marker should be run to ensure that everything is operational on physical Spyre hardware.

## Cutting Releases

See <https://github.com/torch-spyre/sendnn-inference/blob/main/RELEASING.md>
