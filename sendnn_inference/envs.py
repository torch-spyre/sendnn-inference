import os
import platform
from typing import TYPE_CHECKING, Any, Callable

import torch
from vllm.logger import init_logger

from sendnn_inference.utils import parse_cpu_mm_dtype

if TYPE_CHECKING:
    SENDNN_INFERENCE_DYNAMO_BACKEND: str = "sendnn"
    SENDNN_INFERENCE_WARMUP_PROMPT_LENS: list[int] | None = None
    SENDNN_INFERENCE_WARMUP_BATCH_SIZES: list[int] | None = None
    SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED: int = 0
    SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR: str = "/tmp"
    SENDNN_INFERENCE_OVERRIDE_SIGNALS_HANDLER: bool = False
    SENDNN_INFERENCE_CP_INTERLEAVE_STEPS: bool = True
    SENDNN_INFERENCE_UPDATE_THREAD_CONFIG: bool = True
    SENDNN_INFERENCE_MAX_LOAD_PROCESSES: int = 0
    SENDNN_INFERENCE_WORKER_LOG_REDIRECT_DIR: str = ""
    SENDNN_INFERENCE_GLOO_TIMEOUT_MINUTES: int = 60
    SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS: bool = False
    SENDNN_INFERENCE_SIMPLE_COMPILE_BACKEND: str = "inductor"
    SENDNN_INFERENCE_NUM_CPUS: int = 0
    SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG: bool = False
    SENDNN_INFERENCE_MODEL_CONFIG_FILE: str | None = None
    SENDNN_INFERENCE_CPU_MM_DTYPE: torch.dtype = torch.float16

logger = init_logger(__name__)

_cache: dict[str, Any] = {}

_CPU_MM_DTYPE_PLATFORM_DEFAULTS = {"s390x": "float32", "ppc64le": "bfloat16"}


def override(name: str, value: str) -> None:
    if name not in environment_variables:
        raise ValueError(f"The variable {name} is not a known setting and cannot be overridden")
    os.environ[name] = value
    _cache[name] = environment_variables[name]()


def clear_env_cache():
    _cache.clear()


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Defines the prompt lengths the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in pooling.
    "SENDNN_INFERENCE_WARMUP_PROMPT_LENS": lambda: [
        int(p)
        for p in os.getenv(key="SENDNN_INFERENCE_WARMUP_PROMPT_LENS", default="64").split(",")
    ],
    # Defines the batch sizes the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in pooling.
    "SENDNN_INFERENCE_WARMUP_BATCH_SIZES": lambda: [
        int(b) for b in os.getenv(key="SENDNN_INFERENCE_WARMUP_BATCH_SIZES", default="1").split(",")
    ],
    # Defines the backend that torch.compile will use when using Spyre
    # Available options:
    # - "sendnn": Compile for execution on Spyre hardware
    # - "inductor": Compile for execution on CPU (for debug and testing)
    # - "eager": Skip compile entirely (for debug and testing)
    #
    "SENDNN_INFERENCE_DYNAMO_BACKEND": lambda: os.getenv(
        "SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn"
    ),
    # Enable performance metric logging. This captures startup information
    # such as warmup times, and loading times.
    # When `--disable-log-stats=False` is used, this will log timing metrics
    # about every finished request into a .jsonl file. These are the same
    # metrics that are available in prometheus format on the /metrics endpoint,
    # but it is sometime helpful to view them disaggregated to debug performance
    # problems. This logging is not designed to be performant, and should not be
    # enabled in production settings.
    # It is turned off by default.
    "SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED": lambda: int(
        os.getenv("SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED", 0)
    ),
    # Directory to write performance metric logging files. By default,
    # logs are written to /tmp.
    "SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR": lambda: os.getenv(
        "SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR", "/tmp"
    ),
    # If set, override the signal handler for sendnn-inference on
    # vLLM V1 + torch_sendnn backend to be able to gracefully
    # shutdown the engine.
    "SENDNN_INFERENCE_OVERRIDE_SIGNALS_HANDLER": lambda: bool(
        int(os.getenv("SENDNN_INFERENCE_OVERRIDE_SIGNALS_HANDLER", "1"))
    ),
    # Allow sendnn-inference to update env vars related to multi-threading (eg. OMP)
    # based on the detected CPU cores and server configuration
    "SENDNN_INFERENCE_UPDATE_THREAD_CONFIG": lambda: bool(
        int(os.getenv("SENDNN_INFERENCE_UPDATE_THREAD_CONFIG", "1"))
    ),
    # If set, limit the number of concurrent processes loading/compiling
    # large models or models with larger context lengths to limit
    # memory usage.
    # Set to 0 to allow any number of processes
    "SENDNN_INFERENCE_MAX_LOAD_PROCESSES": lambda: int(
        os.getenv("SENDNN_INFERENCE_MAX_LOAD_PROCESSES", "0")
    ),
    # If set, redirects all stdout and stderr from worker processes to files
    # within this director. This is useful for debugging card-specific errors
    # in multi-AIU setups, but should never be enabled in production settings.
    # This removes all output from stdout and stderr for the worker processes.
    "SENDNN_INFERENCE_WORKER_LOG_REDIRECT_DIR": lambda: os.getenv(
        "SENDNN_INFERENCE_WORKER_LOG_REDIRECT_DIR", ""
    ),
    # If set, overrides the default (30 minutes) timeout for
    #  torch.distributed.init_process_group
    "SENDNN_INFERENCE_GLOO_TIMEOUT_MINUTES": lambda: int(
        os.getenv("SENDNN_INFERENCE_GLOO_TIMEOUT_MINUTES", "60")
    ),
    # If set, this will require use of pre-compiled models and
    # disable compilation for decoders
    "SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS": lambda: bool(
        int(os.getenv("SENDNN_INFERENCE_REQUIRE_PRECOMPILED_DECODERS", "0"))
    ),
    # Simple compile backend for some dynamically compiled operations, like
    # gathering logprobs in the sampler.
    # Defaults to eager, iductor can be used if python headers and a compiler
    # are available.
    "SENDNN_INFERENCE_SIMPLE_COMPILE_BACKEND": lambda: os.getenv(
        "SENDNN_INFERENCE_SIMPLE_COMPILE_BACKEND", "inductor"
    ),
    # Configures the number of CPUs used when determining multi-threading
    # configurations
    # Set to 0 to have sendnn-inference attempt to detect the CPU count
    "SENDNN_INFERENCE_NUM_CPUS": lambda: int(os.getenv("SENDNN_INFERENCE_NUM_CPUS", "0")),
    # Feature Flag
    # Works only with chunked prefill enabled. If set, prefill steps are
    # interleaved with a decode step
    "SENDNN_INFERENCE_CP_INTERLEAVE_STEPS": lambda: bool(
        int(os.getenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "1"))
    ),
    # If set, raises a runtime error if the model configuration is not found
    # in the known configurations registry. Only applies when running on
    # Spyre device (sendnn backend).
    "SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG": lambda: bool(
        int(os.getenv("SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG", "0"))
    ),
    # Path to custom model_configs.yaml file. If not set, uses the default
    # location at sendnn_inference/config/model_configs.yaml
    "SENDNN_INFERENCE_MODEL_CONFIG_FILE": lambda: os.getenv("SENDNN_INFERENCE_MODEL_CONFIG_FILE"),
    # Dtype for multimodal vision_tower / multi_modal_projector params (CPU).
    # One of "float32" | "float16" | "bfloat16"; default per platform.
    "SENDNN_INFERENCE_CPU_MM_DTYPE": lambda: parse_cpu_mm_dtype(
        os.getenv(
            "SENDNN_INFERENCE_CPU_MM_DTYPE",
            _CPU_MM_DTYPE_PLATFORM_DEFAULTS.get(platform.machine(), "float16"),
        )
    ),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    if name in _cache:
        return _cache[name]

    # caching and lazy evaluation of environment variables
    if name in environment_variables:
        value = environment_variables[name]()
        _cache[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
