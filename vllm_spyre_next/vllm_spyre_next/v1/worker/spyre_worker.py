"""A Torch Spyre worker class."""

import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.logger import init_logger

from vllm_spyre_next.custom_ops import register_all
from vllm_spyre_next.v1.worker.spyre_model_runner import TorchSpyreModelRunner

logger = init_logger(__name__)


class TorchSpyreWorker(CPUWorker):
    """A worker class that executes the model on IBM's Spyre device.

    Inherits from CPUWorker but overrides init_device to:
    - Skip CPU-specific OMP thread binding
    - Create a TorchSpyreModelRunner with torch.device("spyre")
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the
        # layers will be swapped out with the custom implementations for spyre.
        register_all()

    def init_device(self) -> None:
        # Skip CPU-specific OMP thread binding from CPUWorker.init_device().
        # Spyre execution does not benefit from NUMA-aware thread pinning.
        # If Spyre needs host-side thread affinity in the future, add it here.

        # Distributed environment (reuse upstream logic)
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)

        # Construct Spyre model runner with torch.device("spyre")
        self.model_runner = TorchSpyreModelRunner(
            self.vllm_config,
            torch.device("spyre"),
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is not supported on Spyre, ignoring.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Sleep mode is not supported on Spyre, ignoring.")

    def compile_or_warm_up_model(self):
        # FIXME: Work around for https://github.com/torch-spyre/torch-spyre/issues/1420
        # Ensure registration of Spyre decompositions before FX Graph tracing
        import torch._inductor.decomposition
        from torch_spyre._inductor.decompositions import spyre_decompositions

        for op, impl in spyre_decompositions.items():
            if "addm" in op.name():
                logger.warning(
                    "FIXME: Adding %s decomposition to work-around torch-spyre crash", op.name()
                )
                torch._inductor.decomposition.decompositions[op] = impl
                
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()
        return self.compilation_config.compilation_time
