"""Async vision encoder subprocess for MM pre-encoding.

The encoder process loads only the vision components of the multimodal model
(vision_tower + multi_modal_projector + text_embedding) using FMS's
``get_model(..., vision_only=True)``, which selectively loads ~4 GB of vision
weights from the checkpoint — skipping the LLM decoder entirely.

This process is non-daemon (started by the non-daemon SpyreMultiprocExecutor)
so it runs truly parallel to AIU forward passes.  Results are written to POSIX
shared memory; only a small metadata tuple is sent back through the result queue,
so all TP workers can read the embedding independently without a rank-0 broadcast
of the full tensor.
"""

import logging
import os
import time

import torch
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

import sendnn_inference.envs as envs_spyre
from sendnn_inference.v1.worker.mm_shared_memory import write_embeddings

logger = logging.getLogger(__name__)


def _resolve_mm_utils_cls(hf_config):
    """Return the MMUtils class for *hf_config*.

    MUST be called after ``get_model()`` has run so that FMS has registered
    its adapters — importing ``sendnn_inference.multimodal`` before that
    triggers ``llava_next.py``'s module-level ``extend_adapter`` call, which
    requires the base FMS adapter to already be registered.

    Tries exact class match first (MM_HF_CFG_REGISTRY), then falls back to a
    scan by ``model_type`` string — needed when hf_config is deserialized as the
    base ``PreTrainedConfig`` in the encoder subprocess.
    """
    from sendnn_inference.multimodal import MM_HF_CFG_REGISTRY

    utils_cls = MM_HF_CFG_REGISTRY.get(type(hf_config))
    if utils_cls is not None:
        return utils_cls

    model_type = getattr(hf_config, "model_type", "")

    # Some checkpoints use model_type values that differ from the canonical name
    # stored on the transformers config class (e.g. pixtral checkpoints report
    # model_type="pixtral" but share MM utils with Mistral3Config whose
    # model_type is "mistral3").  Map to the canonical name before scanning.
    _CANONICAL = {"pixtral": "mistral3"}
    canonical = _CANONICAL.get(model_type, model_type)

    for cfg_cls, cls in MM_HF_CFG_REGISTRY.items():
        if getattr(cfg_cls, "model_type", None) == canonical:
            return cls

    raise ValueError(
        f"encoder_process: no MMUtils found for hf_config type={type(hf_config).__name__!r} "
        f"model_type={model_type!r}; known: {[c.__name__ for c in MM_HF_CFG_REGISTRY]}"
    )


# ── VisionEncoderRunner ───────────────────────────────────────────────────────


class VisionEncoderRunner:
    """Loads the vision-only FMS model and encodes MMEncodeRequest jobs.

    Uses ``get_model("hf_pretrained", model_path=..., vision_only=True)`` so
    only the vision tower, projector, and text embedding are loaded from disk.
    """

    def load_model(self, vllm_config: VllmConfig) -> None:
        from fms.models import get_model

        model_config = vllm_config.model_config
        model_path = model_config.model

        if not os.path.isdir(model_path):
            logger.info(
                "encoder_process: %r is not a local directory — resolving from HF cache",
                model_path,
            )
            model_path = download_weights_from_hf(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=["*.safetensors", "*.bin", "*.pt"],
                revision=model_config.revision,
            )

        mm_device = envs_spyre.SENDNN_INFERENCE_MM_DEVICE
        mm_dtype = envs_spyre.SENDNN_INFERENCE_CPU_MM_DTYPE

        logger.info(
            "encoder_process: loading vision-only model from %r (device=%s, dtype=%s)",
            model_path,
            mm_device,
            mm_dtype,
        )
        t0 = time.time()
        self.fms_model = get_model(
            "hf_pretrained",
            model_path=model_path,
            vision_only=True,
        )
        self.fms_model.to(device=mm_device, dtype=mm_dtype).eval()

        # Resolve utils class AFTER get_model() so FMS has registered its adapters.
        # Importing sendnn_inference.multimodal before that triggers llava_next.py's
        # module-level extend_adapter call which requires the base FMS adapter.
        self.mm_utils_cls = _resolve_mm_utils_cls(model_config.hf_config)
        logger.info("encoder_process: mm_utils=%s", self.mm_utils_cls.__name__)
        torch.set_grad_enabled(False)
        self.mm_device = mm_device
        logger.info(
            "encoder_process: vision model loaded in %.2fs", time.time() - t0
        )

    def encode(self, request) -> torch.Tensor:
        """Encode a single MMEncodeRequest and return a CPU-contiguous tensor."""
        input_ids = torch.tensor(
            request.prompt_token_ids, dtype=torch.int64
        ).unsqueeze(0)
        with torch.inference_mode():
            embeds = self.mm_utils_cls.get_maybe_mm_embeddings(
                self.fms_model,
                input_ids,
                request.mm_features,
                is_decode=False,
                mm_device=self.mm_device,
            )
        return embeds.cpu().contiguous()


# ── Process entry point ───────────────────────────────────────────────────────


def encoder_process_main(
    vllm_config: VllmConfig, job_queue, result_queue, stop_event
) -> None:
    """Entry point for the vision encoder subprocess.

    Loads the vision-only model, signals READY, then serves encode jobs.
    Results are written to POSIX SHM; only ``(req_id, shape, dtype)`` metadata
    is put on the result queue so all TP workers can read the embedding
    independently without a rank-0 tensor broadcast.

    ``stop_event`` is a ``multiprocessing.Event`` set by the executor on
    shutdown.  The job loop polls it via a timeout on ``job_queue.get`` so
    the process exits cleanly on both graceful and abrupt server termination.

    Job loop:
      get(MMEncodeRequest) → encode → write SHM → put (req_id, shape, dtype)
    Exits when stop_event is set or None sentinel is received.
    """
    logger.info("encoder_process: starting")

    runner = VisionEncoderRunner()
    try:
        runner.load_model(vllm_config)
    except Exception as exc:
        logger.error(
            "encoder_process: failed to load vision model: %s", exc, exc_info=True
        )
        result_queue.put(f"ERROR: {exc}")
        return

    result_queue.put("READY")
    logger.info("encoder_process: ready, waiting for jobs")

    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("encoder_process: interrupted, exiting")
            break
        except Exception:
            # queue.Empty from timeout — loop back and re-check stop_event.
            continue
        if job is None:
            logger.info("encoder_process: shutdown received")
            break

        req_id = job.request_id
        t0 = time.time()
        logger.info(
            "[parallel-check] encoder_proc: ENCODE START  req='%s'  t=%.3f",
            req_id,
            t0,
        )
        try:
            embeds = runner.encode(job)

            # Write embedding to POSIX SHM; close our handle without unlinking
            # so all TP workers can still open it by name.  Rank 0 will unlink
            # the block after all workers have read (via _collect_async_mm_results).
            shm = write_embeddings(embeds, req_id)
            shm.close()

            t1 = time.time()
            result_queue.put((req_id, tuple(embeds.shape), embeds.dtype))
            logger.info(
                "[parallel-check] encoder_proc: ENCODE END    req='%s'  t=%.3f  "
                "duration=%.2fs",
                req_id,
                t1,
                t1 - t0,
            )
        except Exception as exc:
            logger.error(
                "encoder_process: failed to encode '%s': %s", req_id, exc, exc_info=True
            )
            result_queue.put((req_id, None, None))
