"""This module contains all custom ops for spyre"""

from . import rms_norm
from . import silu_and_mul
from . import vocab_parallel_embedding
from . import linear
from . import rotary_embedding
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_all():
    logger.info("Registering custom ops for spyre_next")
    rms_norm.register()
    silu_and_mul.register()
    vocab_parallel_embedding.register()
    linear.register()
    rotary_embedding.register()
