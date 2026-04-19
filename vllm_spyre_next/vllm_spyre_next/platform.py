import torch
import sys
from typing import TYPE_CHECKING
from string import Template
import multiprocessing
import importlib.metadata


# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get("triton"):
        del sys.modules["triton"]

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

if TYPE_CHECKING:
    # NB: We can't eagerly import many things from vllm since vllm.config
    # will import this file. These would lead to circular imports
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class TorchSpyrePlatform(CpuPlatform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"

    # Primary dispatch key for direct_register_custom_op. Kept as CPU
    # because some custom ops receive CPU-only tensors (e.g. rotary_embedding).
    # All ops are ALSO registered for PrivateUse1 (Spyre) via
    # register_spyre_dispatch() in each module's register() function,
    # so dispatch works regardless of tensor device.
    dispatch_key: str = "CPU"

    # Register the PyTorch Native Attention implementation as the CUSTOM backend
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm_spyre_next.v1.attention.backends.spyre_attn.SpyreAttentionBackend",
    )

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "torch-spyre"

    @classmethod
    def log_server_boot(cls, vllm_config: VllmConfig) -> None:
        # Only log in main process (not in TP workers)
        if multiprocessing.current_process().name != "MainProcess":
            return

        # yapf: disable
        logo_template = Template(
            template="\n       ${w}█     █     █▄   ▄█${r}       ${red}▄█▀▀█▄${r}  ${orange}█▀▀▀█▄${r}  ${yellow}█   █${r}  ${green}█▀▀▀█▄${r}  ${blue}█▀▀▀▀${r}\n" # noqa: E501
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}       ${red}▀▀▄▄▄${r}   ${orange}█▄▄▄█▀${r}  ${yellow}▀▄ ▄▀${r}  ${green}█▄▄▄█▀${r}  ${blue}█▄▄▄${r}   version ${w}%s${r}\n" # noqa: E501
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}            ${red}█${r}  ${orange}█${r}        ${yellow}▀█▀${r}   ${green}█ ▀█▄${r}   ${blue}█${r}      model   ${w}%s${r}\n" # noqa: E501
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}       ${red}▀▄▄▄█▀${r}  ${orange}█${r}         ${yellow}█${r}    ${green}█   ▀█${r}  ${blue}█▄▄▄▄${r}\n" # noqa: E501
        )
        # yapf: enable
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
            "red": "\033[91m",  # red (rainbow start)
            "orange": "\033[38;5;208m",  # orange
            "yellow": "\033[93m",  # yellow
            "green": "\033[92m",  # green
            "blue": "\033[94m",  # blue (rainbow end)
        }

        message = logo_template.substitute(colors)

        version = importlib.metadata.version("vllm_spyre_next")

        model_name = vllm_config.model_config.model if vllm_config.model_config else "N/A"

        logger.info(message, version, model_name)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Set Spyre-specific config defaults before vLLM's defaulting logic."""
        from vllm.config import CompilationMode

        vllm_config.compilation_config.mode = CompilationMode.NONE

        # Force eager execution. torch.compile with the Spyre inductor
        # backend requires ALL graph tensors on Spyre, but our CPU fallback
        # ops (embedding, linear, rotary, attention) create intermediate
        # CPU tensors that the Spyre backend cannot codegen. Once all layers
        # run natively on Spyre, this can be removed to enable compilation.
        vllm_config.model_config.enforce_eager = True

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, *args, **kwargs) -> str:
        if selected_backend == AttentionBackendEnum.CUSTOM:
            return AttentionBackendEnum.CUSTOM.get_path()
        else:
            return super().get_attn_backend_cls(selected_backend, *args, **kwargs)

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls.log_server_boot(vllm_config)

        # Check if the model dtype is different from float16,
        # which is only currently supported in torch-spyre
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError(
                f"The model dtype needs to be torch.float16 for spyre, "
                f"but was specified to be {vllm_config.model_config.dtype}"
            )

        # ---- worker ----
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            # "auto" defaults to the CPUWorker as we inherit from the CpuPlatform
            # Override with TorchSpyreWorker for Spyre-specific functionality
            worker_class = "vllm_spyre_next.v1.worker.spyre_worker.TorchSpyreWorker"
            logger.info("Loading worker from: %s", worker_class)
            parallel_config.worker_cls = worker_class

        # ---- scheduler ----
        scheduler_config = vllm_config.scheduler_config
        # default scheduler
        scheduler_class = "vllm.v1.core.sched.scheduler.Scheduler"
        # if a torch spyre specific scheduler class is needed it can be loaded with
        # scheduler_class = "vllm_spyre_next.v1.core.scheduler.TorchSpyreScheduler"
        logger.info("Loading scheduler from: %s", scheduler_class)
        scheduler_config.scheduler_cls = scheduler_class

        # call CpuPlatform.check_and_update_config()
        super().check_and_update_config(vllm_config)
