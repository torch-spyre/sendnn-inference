import importlib.metadata
import json
from logging.config import dictConfig
from typing import Any

from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH
from vllm.logger import DEFAULT_LOGGING_CONFIG

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
