"""Tests for SpyreCausalLM._cast_params_for_spyre dtype casting."""

from types import SimpleNamespace

import pytest
import torch

from sendnn_inference import envs
from sendnn_inference.model_executor.model_loader.spyre import SpyreCausalLM

pytestmark = pytest.mark.cpu


class _FakeFmsModel:
    """Minimal stand-in exposing only named_parameters()."""

    def __init__(self, specs):
        self._params = [
            (name, torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))) for name, dtype in specs
        ]

    def named_parameters(self):
        return iter(self._params)


class _FakeMMUtils:
    def __init__(self, prefixes):
        self.mm_parameter_prefixes = prefixes


def _cast(fms_model, mm_model_utils):
    SpyreCausalLM._cast_params_for_spyre(
        SimpleNamespace(fms_model=fms_model, mm_model_utils=mm_model_utils)
    )


def _set_cpu_mm_dtype(monkeypatch, value):
    monkeypatch.setenv("SENDNN_INFERENCE_CPU_MM_DTYPE", value)
    envs.clear_env_cache()


def _dtype_of(fms, name):
    return dict(fms.named_parameters())[name].dtype


@pytest.mark.parametrize(
    "initial_dtype,cpu_mm_dtype,expected",
    [
        (torch.bfloat16, "float32", torch.float32),
        (torch.float16, "float16", torch.float16),
    ],
)
def test_mm_params_match_cpu_mm_dtype(monkeypatch, initial_dtype, cpu_mm_dtype, expected):
    _set_cpu_mm_dtype(monkeypatch, cpu_mm_dtype)
    fms = _FakeFmsModel([("vision_tower.weight", initial_dtype)])
    _cast(fms, _FakeMMUtils(("vision_tower.", "multi_modal_projector.")))
    assert _dtype_of(fms, "vision_tower.weight") == expected


def test_bf16_non_mm_params_cast_to_fp16(monkeypatch):
    _set_cpu_mm_dtype(monkeypatch, "float32")
    fms = _FakeFmsModel([("decoder.layers.0.weight", torch.bfloat16)])
    _cast(fms, _FakeMMUtils(("vision_tower.",)))
    assert _dtype_of(fms, "decoder.layers.0.weight") == torch.float16


def test_no_mm_utils_still_casts_bf16_decoder_params(monkeypatch):
    _set_cpu_mm_dtype(monkeypatch, "float32")
    fms = _FakeFmsModel([("decoder.layers.0.weight", torch.bfloat16)])
    _cast(fms, None)
    assert _dtype_of(fms, "decoder.layers.0.weight") == torch.float16


def test_non_bf16_non_mm_param_untouched(monkeypatch):
    _set_cpu_mm_dtype(monkeypatch, "float32")
    fms = _FakeFmsModel([("decoder.layers.0.weight", torch.float32)])
    _cast(fms, _FakeMMUtils(("vision_tower.",)))
    assert _dtype_of(fms, "decoder.layers.0.weight") == torch.float32


def test_mixed_params_all_branches(monkeypatch):
    _set_cpu_mm_dtype(monkeypatch, "float32")
    fms = _FakeFmsModel(
        [
            ("vision_tower.weight", torch.bfloat16),
            ("multi_modal_projector.bias", torch.float32),
            ("decoder.layers.0.weight", torch.bfloat16),
            ("decoder.layers.0.bias", torch.float16),
        ]
    )
    _cast(fms, _FakeMMUtils(("vision_tower.", "multi_modal_projector.")))
    dtypes = {name: p.dtype for name, p in fms.named_parameters()}
    assert dtypes == {
        "vision_tower.weight": torch.float32,
        "multi_modal_projector.bias": torch.float32,
        "decoder.layers.0.weight": torch.float16,
        "decoder.layers.0.bias": torch.float16,
    }
