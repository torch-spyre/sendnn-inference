"""Tests for SpyreCausalLM._cast_params_for_spyre dtype casting."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from sendnn_inference import envs, utils
from sendnn_inference.model_executor.model_loader.spyre import SpyreCausalLM
from sendnn_inference.utils import parse_mm_device

pytestmark = pytest.mark.cpu


class _FakeFmsModel(torch.nn.Module):
    """Minimal real nn.Module stand-in built from dotted parameter paths.

    _cast_params_for_spyre now casts the whole model with Module.to(...) and
    places the multimodal submodules via named_modules(), so the fake has to be a
    genuine module tree (supporting .to() and named_modules()), not just a
    named_parameters() shim. A spec entry "vision_tower.weight" creates a
    submodule "vision_tower" holding a parameter "weight".
    """

    def __init__(self, specs):
        super().__init__()
        for path, dtype in specs:
            *mod_parts, param_name = path.split(".")
            parent = self
            for part in mod_parts:
                child = parent._modules.get(part)
                if child is None:
                    child = torch.nn.Module()
                    parent.add_module(part, child)
                parent = child
            parent.register_parameter(
                param_name, torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))
            )


class _FakeMMUtils:
    def __init__(self, prefixes):
        self.mm_parameter_prefixes = prefixes


def _cast(fms_model, mm_model_utils):
    SpyreCausalLM._cast_params_for_spyre(
        SimpleNamespace(fms_model=fms_model, mm_model_utils=mm_model_utils)
    )


def _set_cpu_mm_dtype(monkeypatch, value):
    monkeypatch.setenv("SENDNN_INFERENCE_CPU_MM_DTYPE", value)
    # Keep the multimodal device on CPU so these dtype-focused tests don't depend
    # on torch_nnpa being installed / an nnpa device being usable.
    monkeypatch.setenv("SENDNN_INFERENCE_MM_DEVICE", "cpu")
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


def test_non_bf16_non_mm_param__untouched(monkeypatch):
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
        # mm submodules -> cpu_mm_dtype (float32), overriding the whole-model cast
        "vision_tower.weight": torch.float32,
        "multi_modal_projector.bias": torch.float32,
        # everything else -> fp16
        "decoder.layers.0.weight": torch.float16,
        "decoder.layers.0.bias": torch.float16,
    }


def _force_nnpa_detected(monkeypatch, detected):
    """Make parse_mm_device see torch_nnpa as present/absent without installing
    it, by stubbing the find_spec it uses. Leaves other imports untouched."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torch_nnpa":
            return object() if detected else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def _set_mm_device(monkeypatch, value):
    monkeypatch.setenv("SENDNN_INFERENCE_CPU_MM_DTYPE", "float16")
    monkeypatch.setenv("SENDNN_INFERENCE_MM_DEVICE", value)
    envs.clear_env_cache()


def _cast_returning_self(fms_model, mm_model_utils):
    ns = SimpleNamespace(fms_model=fms_model, mm_model_utils=mm_model_utils)
    SpyreCausalLM._cast_params_for_spyre(ns)
    return ns


def test_parse_mm_device_cpu(monkeypatch):
    _force_nnpa_detected(monkeypatch, False)
    assert parse_mm_device("cpu") == "cpu"


def test_parse_mm_device_auto_without_nnpa_resolves_cpu(monkeypatch):
    _force_nnpa_detected(monkeypatch, False)
    assert parse_mm_device("auto") == "cpu"


def test_parse_mm_device_auto_with_nnpa_resolves_nnpa(monkeypatch):
    _force_nnpa_detected(monkeypatch, True)
    assert parse_mm_device("auto") == "nnpa"


def test_parse_mm_device_explicit_nnpa_missing_raises(monkeypatch):
    _force_nnpa_detected(monkeypatch, False)
    with pytest.raises(ImportError):
        parse_mm_device("nnpa")


def test_nnpa_registration_failure_raises(monkeypatch):
    # torch_nnpa is importable (detected) so MM_DEVICE resolves to "nnpa", but
    # the device can't be initialized -> hard error, no silent CPU fallback.
    _force_nnpa_detected(monkeypatch, True)
    _set_mm_device(monkeypatch, "nnpa")
    monkeypatch.setattr(utils, "ensure_nnpa_registered", lambda: False)
    fms = _FakeFmsModel([("vision_tower.weight", torch.bfloat16)])
    with pytest.raises(RuntimeError, match="nnpa device could not be initialized"):
        _cast(fms, _FakeMMUtils(("vision_tower.",)))


def test_nnpa_text_only_model_never_registers_or_errors(monkeypatch):
    # A text-only model (no mm submodules) must not attempt nnpa registration
    # and must not error even when MM_DEVICE resolves to "nnpa".
    _force_nnpa_detected(monkeypatch, True)
    _set_mm_device(monkeypatch, "nnpa")
    calls = []
    monkeypatch.setattr(utils, "ensure_nnpa_registered", lambda: calls.append(True) or False)
    fms = _FakeFmsModel([("decoder.layers.0.weight", torch.bfloat16)])
    ns = _cast_returning_self(fms, None)
    assert calls == []  # registration never attempted (short-circuited)
    assert ns.mm_device == "nnpa"
    assert _dtype_of(fms, "decoder.layers.0.weight") == torch.float16
