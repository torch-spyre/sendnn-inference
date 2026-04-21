# Installation

We use the [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies. `uv` provides advanced
dependency resolution which is required to properly install dependencies like
`vllm` without overwriting critical dependencies like `torch`.

## Install `uv`

You can [install `uv`](https://docs.astral.sh/uv/guides/install-python/) using `pip`:

```sh
pip install uv
```

## Create a Python Virtual Environment

Now create and activate a new Python (3.12) [virtual environment](https://docs.astral.sh/uv/pip/environments/):

```sh
uv venv --python 3.12 --seed .venv --system-site-packages
source .venv/bin/activate
```

??? question "Why do I want the `--system-site-packages`?"
    Because the full `torch_sendnn` stack is only available pre-installed in a
    base environment, we need to add the `--system-site-packages` to the new
    virtual environment in order to fully support the Spyre hardware.

    **Note**, pulling in the system site packages is not required for CPU-only
    installations.

## Install vLLM with the SenDNN Inference Plugin

You can either install a released version of the SenDNN Inference plugin directly from
[PyPI](https://pypi.org/project/sendnn-inference/) or you can install from source by
cloning the [SenDNN Inference](https://github.com/torch-spyre/sendnn-inference) repo from
GitHub.

=== "Release (PyPI)"

    ```sh
    uv pip install sendnn-inference
    ```

=== "Source (GitHub)"

    First, clone the `sendnn-inference` repo:
    
    ```sh
    git clone https://github.com/torch-spyre/sendnn-inference.git
    cd sendnn-inference
    ```
    
    To install `sendnn-inference` locally with development dependencies, use the following command:
    
    ```sh
    uv sync --frozen --active --inexact
    ```
    
    !!! tip
        The `dev` group (i.e. `--group dev`) is enabled by default.

## Install PyTorch

Finally, `torch` is needed to run examples and tests. If it is not already installed,
install it using `pip`.

The Spyre runtime stack supports specific `torch` versions. Use the compatible version for each `torch_sendnn` release:

| torch_sendnn | torch |
| -- | -- |
| 1.0.0 | 2.7.1 |
| 1.1.0 | 2.7.1 |
| 1.2.0 | 2.10.0 |

=== "Linux"

    ```sh
    pip install torch=="2.10.0+cpu" --index-url "https://download.pytorch.org/whl/cpu"
    ```

=== "Windows/macOS"

    ```sh
    pip install torch=="2.10.0"
    ```

!!! note
    On Linux the `+cpu` package should be installed, since we don't need any of
    the `cuda` dependencies which are included by default for Linux installs.
    This requires `--index-url https://download.pytorch.org/whl/cpu` on Linux.
    On Windows and macOS the CPU package is the default one.

## Troubleshooting

As the installation process is evolving over time, you may have arrived here after
following outdated installation steps. If you encountered any of the errors below,
it may be easiest to start over with a new Python virtual environment (`.venv`)
as outlined above.

### Installation using `pip` (instead of `uv`)

If you happen to have followed the pre-`uv` installation instructions, you might
encounter an error like this:

```sh
LookupError: setuptools-scm was unable to detect version for /home/senuser/multi-aiu-dev/_dev/sentient-ci-cd/_dev/sen_latest/sendnn-inference.
      
    Make sure you're either building from a fully intact git repository or PyPI tarballs. Most other sources (such as GitHub's tarballs, a git checkout without the .git folder) don't contain the necessary metadata and will not work.
      
    For example, if you're using pip, instead of https://github.com/user/proj/archive/master.zip use git+https://github.com/user/proj.git#egg=proj
```

Make sure the follow the latest installation steps outlined above.

### Failed to activate the Virtual Environment

If you encounter any of the following errors, it's likely you forgot to activate
the (correct) Python Virtual Environment:

```sh
  File "/home/senuser/.local/lib/python3.12/site-packages/vllm/config.py", line 2260, in __post_init__
    self.device = torch.device(self.device_type)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Device string must not be empty
```

### No module named `torch`

You may have installed PyTorch into the system-wide Python environment, not into
the virtual environment used for SenDNN Inference:

```sh
  File "/home/senuser/multi-aiu-dev/_dev/sentient-ci-cd/_dev/sen_latest/sendnn-inference/.venv/lib64/python3.12/site-packages/vllm/env_override.py", line 4, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

Make sure to activate the same virtual environment for installing `torch` that
was used to install `sendnn-inference`. If you already have a system-wide `torch`
installation and want to reuse that for your `sendnn-inference` environment, you can
create a new virtual environment and add the `--system-site-packages` flag to
pull in the `torch` dependencies from the base Python environment:

```sh
rm -rf .venv
uv venv --python 3.12 --seed .venv --system-site-packages
source .venv/bin/activate
```

### No solution found when resolving dependencies (Legacy)

!!! note
    This error should no longer occur with torch 2.10.0 support. This section is kept for reference.

If you forget to override the `torch` dependencies when installing from PyPI you
will likely see a dependency resolution error like this:

```sh
$ uv pip install sendnn-inference==0.4.1
  ...
  × No solution found when resolving dependencies:
  ╰─▶ Because fms-model-optimizer==0.2.0 depends on torch>=2.1,<2.5 and only the following versions of fms-model-optimizer are available:
          fms-model-optimizer<=0.2.0
          fms-model-optimizer==0.3.0
      we can conclude that fms-model-optimizer<0.3.0 depends on torch>=2.1,<2.5.
      And because fms-model-optimizer==0.3.0 depends on torch>=2.2.0,<2.6 and all of:
          vllm>=0.9.0,<=0.9.0.1
          vllm>=0.9.2
      depend on torch==2.7.0, we can conclude that all versions of fms-model-optimizer and all of:
          vllm>=0.9.0,<=0.9.0.1
          vllm>=0.9.2
       are incompatible.
      And because only the following versions of vllm are available:
          vllm<=0.9.0
          vllm==0.9.0.1
          vllm==0.9.1
          vllm==0.9.2
      and sendnn-inference==0.4.1 depends on fms-model-optimizer, we can conclude that all of:
          vllm>=0.9.0,<0.9.1
          vllm>0.9.1
       and sendnn-inference==0.4.1 are incompatible.
      And because sendnn-inference==0.4.1 depends on one of:
          vllm>=0.9.0,<0.9.1
          vllm>0.9.1
      and you require sendnn-inference==0.4.1, we can conclude that your requirements are unsatisfiable.
```

<!-- markdownlint-disable MD051 link-fragments -->

To avoid this error, make sure to include the dependency `--overrides` as described
in the installation from a [Release (PyPI)](#release-pypi) section.

<!-- markdownlint-enable MD051 -->
