import contextlib
import importlib.util
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


_CPU_MM_DTYPES = ("float32", "float16", "bfloat16")


def parse_cpu_mm_dtype(value: str) -> torch.dtype:
    if value not in _CPU_MM_DTYPES:
        raise ValueError(
            f"SENDNN_INFERENCE_CPU_MM_DTYPE must be one of {list(_CPU_MM_DTYPES)}, got {value!r}"
        )
    return getattr(torch, value)


_MM_DEVICES = ("auto", "cpu", "nnpa")


def parse_mm_device(value: str) -> str:
    """Validate SENDNN_INFERENCE_MM_DEVICE and return the *intended* device name
    ("cpu" or "nnpa").

    This only resolves intent; it deliberately does NOT import torch_nnpa or
    otherwise touch the torch privateuse1 backend. Renaming privateuse1 before
    the distributed environment is initialized breaks CPU/gloo collectives. The
    actual torch_nnpa registration is deferred to multimodal weight placement --
    see ensure_nnpa_registered().

    Accepted input values:
      - "auto" (default): use "nnpa" if torch_nnpa is importable, else "cpu".
      - "cpu": force CPU execution.
      - "nnpa": force NNPA; raise ImportError if torch_nnpa is missing.
    """
    value = value.lower()
    if value not in _MM_DEVICES:
        raise ValueError(
            f"SENDNN_INFERENCE_MM_DEVICE must be one of {list(_MM_DEVICES)}, got {value!r}"
        )
    if value == "cpu":
        return "cpu"

    # find_spec walks sys.path WITHOUT importing torch_nnpa, so the privateuse1
    # backend is left untouched until ensure_nnpa_registered() runs.
    nnpa_available = importlib.util.find_spec("torch_nnpa") is not None
    if value == "nnpa":
        if not nnpa_available:
            raise ImportError(
                "SENDNN_INFERENCE_MM_DEVICE=nnpa requires the torch_nnpa package; "
                "install it to run multimodal vision encoders on the NNPA device."
            )
        return "nnpa"
    # value == "auto"
    if nnpa_available:
        logger.info("torch_nnpa detected; multimodal vision tower will target nnpa.")
        return "nnpa"
    logger.debug("torch_nnpa not available; multimodal vision tower will run on CPU.")
    return "cpu"


_nnpa_registered: bool | None = None


def ensure_nnpa_registered() -> bool:
    """Fully register the torch_nnpa privateuse1 backend (including the
    PrivateUse1HooksInterface) and return whether the nnpa device is usable.

    Importing torch_nnpa renames the torch privateuse1 backend to "nnpa", but the
    C++ PrivateUse1HooksInterface is normally registered by PyTorch's device
    backend autoload (TORCH_DEVICE_BACKEND_AUTOLOAD). When autoload is disabled
    (=0), a bare ``import torch_nnpa`` leaves the backend half-registered, and any
    subsequent CPU/gloo ``torch.distributed`` collective crashes in
    ``torch._C._get_accelerator()`` with "register PrivateUse1HooksInterface
    first". This helper replicates what autoload does by invoking torch_nnpa's
    ``torch.backends`` entry point explicitly, then verifies the device actually
    works. The result is cached so repeated calls are cheap.

    Returns True if nnpa is usable; False otherwise (callers should fall back to
    CPU for the multimodal vision tower).
    """
    global _nnpa_registered
    if _nnpa_registered is None:
        _nnpa_registered = _register_nnpa()
    return _nnpa_registered


def _register_nnpa() -> bool:
    try:
        # torch_nnpa is only distributed for s390x. CI runs on x86_64,
        # so this optional import is intentionally unresolved there.
        import torch_nnpa  # ty: ignore[unresolved-import]

        logger.info(
            "torch_nnpa %s detected; nnpa device available=%s.",
            torch_nnpa.__version__,
            torch_nnpa.NNPAModule.is_available(),
        )
    except ImportError:
        logger.warning("torch_nnpa is not installed; multimodal vision tower will run on CPU.")
        return False

    # Replicate PyTorch's device-backend autoload, which is what registers the
    # PrivateUse1HooksInterface. Required because the deployment runs with
    # TORCH_DEVICE_BACKEND_AUTOLOAD=0, so importing torch_nnpa above only renames
    # the backend without registering the hooks interface.
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="torch.backends"):
            name = (ep.name or "").lower()
            value = (ep.value or "").lower()
            if "nnpa" in name or value.startswith("torch_nnpa"):
                ep.load()()
    except Exception as e:  # noqa: BLE001 - best effort; usability is verified below
        logger.warning("Failed to run torch_nnpa backend autoload entry point: %s", e)

    # Verify the device is actually usable before committing to it. If the hooks
    # interface still isn't registered, this raises and we fall back to CPU.
    try:
        torch.zeros(1, device="nnpa")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "torch_nnpa is installed but the nnpa device is not usable (%s); "
            "multimodal vision tower will run on CPU.",
            e,
        )
        return False

    logger.info("torch_nnpa registered; multimodal vision tower will run on nnpa.")
    return True
