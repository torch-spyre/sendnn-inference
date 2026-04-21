import contextlib
import math

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def stagger_region(limit: int, world_size: int, rank: int):
    """
    Limit the number of concurrent processes into this region of code.
    Processes yield from this function when they are allowed to enter the
    region of code. Processes return from this function when all of the
    processes have completed the region of code.

    :param limit: Number of concurrent processes allowed if > 0.
    :param world_size: Total world size, usually TP * PP.
    :param rank: Rank of calling worker process.
    """
    if limit > 0 and limit < world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank < (_set + 1) * limit:
                break
            torch.distributed.barrier()
        logger.info(
            "Stagger Region Enter (Set: %d) of %d", _set + 1, math.ceil(world_size / float(limit))
        )
    yield {}

    # TODO: make sure this isn't called excessively

    if limit > 0 and limit < world_size:
        logger.info("Rank %d Done With Stagger Region", rank)
        for _set in range(math.ceil(world_size / float(limit))):
            if rank >= (_set + 1) * limit:
                continue
            torch.distributed.barrier()
        logger.info("Stagger Region: All Complete")


def exact_div(a: int, b: int) -> int:
    q, r = divmod(a, b)
    if r != 0:
        raise ValueError(f"{a} is not exactly divisible by {b}")
    return q
