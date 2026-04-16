"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Shared registry: layer_name -> layer instance (for custom op lookup)
_LAYER_REGISTRY: dict[str, Any] = {}
_INSTANCE_COUNTERS: dict[str, int] = {}


def register_layer(instance: Any, prefix: str) -> str:
    """Register a layer instance and return its unique name.

    Used by custom ops that need to look up `self` from a standalone
    function (the custom op runs outside torch.compile and receives
    only a string key).

    Args:
        instance: The layer instance to register.
        prefix: Base name, e.g. "spyre_rmsnorm".

    Returns:
        Unique layer name, e.g. "spyre_rmsnorm_0".
    """
    count = _INSTANCE_COUNTERS.get(prefix, 0)
    name = f"{prefix}_{count}"
    _INSTANCE_COUNTERS[prefix] = count + 1
    _LAYER_REGISTRY[name] = instance
    return name


def get_layer(name: str) -> Any:
    """Look up a registered layer by name."""
    return _LAYER_REGISTRY[name]


def _fake_impl(*args, **kwargs) -> None:
    """No-op fake implementation for shape inference during torch.compile tracing."""
    return


# Library object for registering PrivateUse1 (Spyre) dispatch keys.
# vLLM's direct_register_custom_op only registers the platform dispatch_key
# (CPU). When tensors flow on Spyre, PyTorch dispatches to PrivateUse1 —
# so we must register the same kernels there too.
_spyre_dispatch_lib = torch.library.Library("vllm", "IMPL")


def register_spyre_dispatch(op_name: str, op_func) -> None:
    """Register an op implementation for the PrivateUse1 (Spyre) dispatch key.

    Call this after direct_register_custom_op to ensure the op can be
    dispatched when any argument is a Spyre tensor.
    """
    _spyre_dispatch_lib.impl(op_name, op_func, dispatch_key="PrivateUse1")


def convert(tensor, device=None, dtype=None):
    """Convert tensor device and/or dtype. No-op when both are None.

    Args:
        tensor: Input tensor, or None (passed through as None).
        device: Target device (None = keep current).
        dtype: Target dtype (None = keep current).

    Returns:
        Converted tensor, or None if input is None.
    """
    if tensor is None:
        return None

    target_device = device if device is not None else tensor.device.type
    target_dtype = dtype if dtype is not None else tensor.dtype

    if tensor.device.type == target_device and tensor.dtype == target_dtype:
        return tensor

    # Spyre requires CPU for dtype changes
    if tensor.device.type == "spyre" and tensor.dtype != target_dtype:
        tensor = tensor.to(device="cpu")

    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)

    if tensor.device.type != target_device:
        tensor = tensor.to(device=target_device)

    return tensor
