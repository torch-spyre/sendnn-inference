"""Tests for our environment configs"""

import os

import pytest

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
