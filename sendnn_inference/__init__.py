import importlib.util
import os

# Disable PyTorch's device-backend autoload, but ONLY when torch_nnpa is
# installed -- it's torch_nnpa's autoload that breaks worker startup, so we don't
# change torch's default behaviour on hosts without it. find_spec checks
# availability without importing torch_nnpa (or torch), leaving the privateuse1
# backend untouched. This MUST run before torch is imported (autoload fires
# during `import torch`).
#
# With autoload enabled (the default, "1"), `import torch` eagerly imports every
# package that registers a `torch.backends` entry point -- here torch_nnpa --
# which renames the privateuse1 backend so torch._C._get_accelerator() resolves
# to PrivateUse1. In a spawned vLLM worker that backend's
# PrivateUse1HooksInterface isn't registered at that point, so the first
# CPU/gloo collective in vLLM's init_distributed_environment
# (_node_count -> torch.distributed.barrier) crashes with
# "register PrivateUse1HooksInterface first".
#
# We register the backends we actually need explicitly and lazily instead:
# torch_sendnn via SpyrePlatform.maybe_ensure_sendnn_configured, and torch_nnpa
# (for the multimodal vision tower) via utils.ensure_nnpa_registered. setdefault
# lets an explicit operator override win. Spawned workers inherit this value
# from the engine process that imports this package during plugin loading, so it
# is in their environment before they `import torch`. The user does not need to
# set TORCH_DEVICE_BACKEND_AUTOLOAD or SENDNN_INFERENCE_MM_DEVICE manually:
# autoload is handled here, and SENDNN_INFERENCE_MM_DEVICE defaults to "auto"
# (use nnpa when available, else CPU).
if importlib.util.find_spec("torch_nnpa") is not None:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import importlib.metadata  # noqa: E402
import json  # noqa: E402
from logging.config import dictConfig  # noqa: E402
from typing import Any  # noqa: E402

from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH  # noqa: E402
from vllm.logger import DEFAULT_LOGGING_CONFIG  # noqa: E402

__version__ = importlib.metadata.version("sendnn_inference")


def register():
    """Register the Spyre platform."""
    return "sendnn_inference.platform.SpyrePlatform"


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config: dict[str, Any] = {}

    if VLLM_CONFIGURE_LOGGING:
        config = {**DEFAULT_LOGGING_CONFIG}

    if VLLM_LOGGING_CONFIG_PATH:
        # Error checks must be done already in vllm.logger.py
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            config = json.loads(file.read())

    if VLLM_CONFIGURE_LOGGING:
        # Copy the vLLM logging configurations for our package
        if "sendnn_inference" not in config["formatters"]:
            if "vllm" in config["formatters"]:
                config["formatters"]["sendnn_inference"] = config["formatters"]["vllm"]
            else:
                config["formatters"]["sendnn_inference"] = DEFAULT_LOGGING_CONFIG["formatters"][
                    "vllm"
                ]

        if "sendnn_inference" not in config["handlers"]:
            if "vllm" in config["handlers"]:
                handler_config = config["handlers"]["vllm"]
            else:
                handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
            handler_config["formatter"] = "sendnn_inference"
            config["handlers"]["sendnn_inference"] = handler_config

        if "sendnn_inference" not in config["loggers"]:
            if "vllm" in config["loggers"]:
                logger_config = config["loggers"]["vllm"]
            else:
                logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
            logger_config["handlers"] = ["sendnn_inference"]
            config["loggers"]["sendnn_inference"] = logger_config

    dictConfig(config)


_init_logging()
