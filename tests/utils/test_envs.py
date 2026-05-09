"""Tests for our environment configs"""

import os

import pytest
import torch

from sendnn_inference import envs

pytestmark = pytest.mark.cpu


def test_env_vars_are_cached(monkeypatch):
    monkeypatch.setenv("SENDNN_INFERENCE_NUM_CPUS", "42")
    assert envs.SENDNN_INFERENCE_NUM_CPUS == 42

    # Future reads don't query the environment every time, so this should not
    # return the updated value
    monkeypatch.setenv("SENDNN_INFERENCE_NUM_CPUS", "77")
    assert envs.SENDNN_INFERENCE_NUM_CPUS == 42


def test_env_vars_override(monkeypatch):
    monkeypatch.setenv("SENDNN_INFERENCE_NUM_CPUS", "42")
    assert envs.SENDNN_INFERENCE_NUM_CPUS == 42

    # This override both sets the environment variable and updates our cache
    envs.override("SENDNN_INFERENCE_NUM_CPUS", "77")
    assert envs.SENDNN_INFERENCE_NUM_CPUS == 77
    assert os.getenv("SENDNN_INFERENCE_NUM_CPUS") == "77"


def test_env_vars_override_with_bad_value(monkeypatch):
    monkeypatch.setenv("SENDNN_INFERENCE_NUM_CPUS", "42")
    assert envs.SENDNN_INFERENCE_NUM_CPUS == 42

    # envs.override ensures the value can be parsed correctly
    with pytest.raises(ValueError, match=r"invalid literal for int"):
        envs.override("SENDNN_INFERENCE_NUM_CPUS", "notanumber")


def test_env_vars_override_for_invalid_config():
    with pytest.raises(ValueError, match=r"not a known setting"):
        envs.override("SENDNN_INFERENCE_NOT_A_CONFIG", "nothing")


@pytest.mark.parametrize(
    "machine,expected",
    [
        ("s390x", torch.float32),
        ("ppc64le", torch.bfloat16),
        ("x86_64", torch.float16),
        ("aarch64", torch.float16),
    ],
)
def test_cpu_mm_dtype_platform_default(monkeypatch, machine, expected):
    monkeypatch.delenv("SENDNN_INFERENCE_CPU_MM_DTYPE", raising=False)
    monkeypatch.setattr("platform.machine", lambda: machine)
    envs.clear_env_cache()
    assert expected == envs.SENDNN_INFERENCE_CPU_MM_DTYPE


def test_cpu_mm_dtype_env_var_wins_over_platform_default(monkeypatch):
    monkeypatch.setenv("SENDNN_INFERENCE_CPU_MM_DTYPE", "bfloat16")
    monkeypatch.setattr("platform.machine", lambda: "s390x")
    envs.clear_env_cache()
    assert torch.bfloat16 == envs.SENDNN_INFERENCE_CPU_MM_DTYPE


def test_cpu_mm_dtype_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("SENDNN_INFERENCE_CPU_MM_DTYPE", "foo")
    envs.clear_env_cache()
    with pytest.raises(ValueError, match=r"SENDNN_INFERENCE_CPU_MM_DTYPE must be one of"):
        _ = envs.SENDNN_INFERENCE_CPU_MM_DTYPE
