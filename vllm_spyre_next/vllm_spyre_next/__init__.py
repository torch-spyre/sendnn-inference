import importlib.metadata
import json
from logging.config import dictConfig
from typing import Any

from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH
from vllm.logger import DEFAULT_LOGGING_CONFIG

__version__ = importlib.metadata.version("vllm_spyre_next")


def register():
    """Register the Spyre platform."""
    return "vllm_spyre_next.platform.TorchSpyrePlatform"


def register_ops():
    """Register OOT custom ops for Spyre."""
    from vllm_spyre_next.custom_ops import register_all

    register_all()


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config: dict[str, Any] = {}

    if VLLM_CONFIGURE_LOGGING:
        config = {**DEFAULT_LOGGING_CONFIG}

    if VLLM_LOGGING_CONFIG_PATH:
        # Error checks must already be done in vllm.logger
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            config = json.loads(file.read())

    if VLLM_CONFIGURE_LOGGING:
        # Copy the vLLM logging configurations for our package
        if "vllm_spyre_next" not in config["formatters"]:
            if "vllm" in config["formatters"]:
                config["formatters"]["vllm_spyre_next"] = config["formatters"]["vllm"]
            else:
                config["formatters"]["vllm_spyre_next"] = DEFAULT_LOGGING_CONFIG["formatters"][
                    "vllm"
                ]

        if "vllm_spyre_next" not in config["handlers"]:
            if "vllm" in config["handlers"]:
                handler_config = config["handlers"]["vllm"]
            else:
                handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
            handler_config["formatter"] = "vllm_spyre_next"
            config["handlers"]["vllm_spyre_next"] = handler_config

        if "vllm_spyre_next" not in config["loggers"]:
            if "vllm" in config["loggers"]:
                logger_config = config["loggers"]["vllm"]
            else:
                logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
            logger_config["handlers"] = ["vllm_spyre_next"]
            config["loggers"]["vllm_spyre_next"] = logger_config

    dictConfig(config)


_init_logging()
