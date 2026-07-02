"""Utilities for sharing multimodal embeddings across TP ranks via POSIX shared memory.

During chunked prefill rank 0 computes the full vision-encoder embeddings and writes
them here; non-zero ranks read after synchronisation in the model runner.
This avoids running the (CPU-bound) vision encoder world_size times per request.
"""

import hashlib
import math
from multiprocessing.shared_memory import SharedMemory

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Stable mapping between torch dtypes and compact integer identifiers used in
# the broadcast metadata tensor.  torch.frombuffer handles all dtypes natively.
_DTYPE_TO_IDX: dict[torch.dtype, int] = {
    torch.float16: 0,
    torch.float32: 1,
    torch.bfloat16: 2,
}
_IDX_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_IDX.items()}


def dtype_to_idx(dtype: torch.dtype) -> int:
    """Encode a torch dtype as a compact integer for the broadcast metadata tensor."""
    if dtype not in _DTYPE_TO_IDX:
        raise ValueError(f"Unsupported dtype for SHM transfer: {dtype}")
    return _DTYPE_TO_IDX[dtype]


def idx_to_dtype(idx: int) -> torch.dtype:
    """Decode a compact integer back to the corresponding torch dtype."""
    if idx not in _IDX_TO_DTYPE:
        raise ValueError(f"Unknown dtype index: {idx}")
    return _IDX_TO_DTYPE[idx]


def _shm_name(req_id: str) -> str:
    """Generate a short, deterministic POSIX SHM name for a request.

    Uses an MD5 hash of the *full* req_id so that requests which share a
    common prefix (e.g. all benchmark requests in a run share the
    ``chatcmpl-bench-<uuid>-`` prefix) still get distinct SHM names.

    Truncating the req_id (the previous approach) caused silent collisions:
    ``chatcmpl-bench-34e3ed2d-1-…`` and ``chatcmpl-bench-34e3ed2d-39-…``
    both hash to the same 20-char prefix, so every request in the benchmark
    wrote to the same SHM segment — corrupting each other's embeddings.

    Linux NAME_MAX is 255; macOS requires ≤ 30 chars for the name itself
    (the kernel prefixes it with ``/``).  'sm' + 16 hex chars = 18 chars,
    safely within every platform's limit.
    """
    digest = hashlib.md5(req_id.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"sm{digest}"


def write_embeddings(tensor: torch.Tensor, req_id: str) -> SharedMemory:
    """Write *tensor* to a shared-memory block keyed by *req_id*.

    Returns the ``SharedMemory`` handle — the caller must keep it and pass it
    to :func:`cleanup_embeddings` after all ranks have read.

    Shape and dtype are NOT stored in SHM; the caller broadcasts them via a
    tiny ``torch.distributed.broadcast`` so readers already have that info
    before calling :func:`read_embeddings`.
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.contiguous()

    assert tensor.ndim == 3, f"Expected 3-D embedding tensor, got shape {tensor.shape}"
    assert tensor.dtype in _DTYPE_TO_IDX, f"Unsupported dtype for SHM transfer: {tensor.dtype}"

    data_shm = SharedMemory(create=True, size=tensor.nbytes, name=_shm_name(req_id))
    # POSIX shared memory rounds the segment up to a page boundary (16 KiB
    # on macOS ARM64, 4 KiB on Linux x86_64), so data_shm.buf is generally
    # larger than tensor.nbytes. Slice to the exact byte count before
    # reshaping, otherwise the trailing padding produces an off-by-page-size
    # reshape error.
    torch.frombuffer(data_shm.buf[: tensor.nbytes], dtype=tensor.dtype).reshape(
        tensor.shape
    ).copy_(tensor)

    logger.debug(
        "Wrote MM embeddings to SHM for req '%s': shape=%s dtype=%s bytes=%d",
        req_id,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.nbytes,
    )
    return data_shm


def read_embeddings(
    req_id: str,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read embeddings from shared memory and return a detached CPU tensor.

    *shape* and *dtype* must be provided by the caller (obtained from the
    broadcast metadata tensor) — they are not re-read from SHM.

    Opens and closes the shared-memory handle internally.
    """
    data_shm = SharedMemory(name=_shm_name(req_id))
    # Match the byte count slice from write_embeddings — the SHM buf is
    # page-padded, so passing the full buf would fail .reshape(shape).
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    # .clone() detaches the tensor from the SHM buffer so the handle can be closed.
    result = torch.frombuffer(data_shm.buf[:nbytes], dtype=dtype).reshape(shape).clone()
    data_shm.close()

    logger.debug(
        "Read MM embeddings from SHM for req '%s': shape=%s dtype=%s",
        req_id,
        shape,
        dtype,
    )
    return result


def cleanup_embeddings(data_shm: SharedMemory) -> None:
    """Unlink and close the shared-memory block.

    Safe to call even if the block was already cleaned up — exceptions are
    logged but not re-raised.
    """
    try:
        data_shm.unlink()
    except Exception as exc:
        logger.debug("SHM unlink skipped (%s): %s", data_shm.name, exc)
    try:
        data_shm.close()
    except Exception as exc:
        logger.debug("SHM close skipped (%s): %s", data_shm.name, exc)


def cleanup_embeddings_by_name(req_id: str) -> None:
    """Unlink the shared-memory block for *req_id* by name.

    Used when the caller does not hold the original ``SharedMemory`` handle
    (e.g. after all TP ranks have independently read the block and rank 0
    needs to unlink it).  Safe to call even if the block was already removed.
    """
    name = _shm_name(req_id)
    try:
        shm = SharedMemory(name=name)
        shm.unlink()
        shm.close()
    except Exception as exc:
        logger.debug("SHM cleanup_by_name skipped (%s): %s", name, exc)
