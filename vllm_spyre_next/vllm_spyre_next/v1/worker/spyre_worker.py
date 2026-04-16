"""A Torch Spyre worker class."""

from vllm.config import VllmConfig
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.logger import init_logger

from vllm_spyre_next.custom_ops import register_all

logger = init_logger(__name__)


class TorchSpyreWorker(CPUWorker):
    """A worker class that executes the model on a group of Spyre cores."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the layers will be swapped out
        # with the custom implementations for spyre.
        register_all()

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
        return super().compile_or_warm_up_model()
