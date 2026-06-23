"""Async vision encoder subprocess for MM pre-encoding.

The encoder process loads only the vision components of the multimodal model
(vision_tower + multi_modal_projector + text_embedding) using FMS's
``get_model(..., vision_only=True)``, which selectively loads vision
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

import sendnn_inference.envs as envs_spyre
from sendnn_inference.model_executor.model_loader.spyre import SpyreCausalLM, cast_params_for_spyre
from sendnn_inference.platform import SpyrePlatform
from sendnn_inference.v1.worker.mm_shared_memory import write_embeddings

logger = logging.getLogger(__name__)


def _resolve_mm_utils_cls(hf_config):
    """Return the MMUtils class for *hf_config*.

    Callers should first pass *hf_config* through
    ``SpyreCausalLM.resolve_hf_config()`` so that format-specific conversions
    (e.g. Mistral-format pixtral → Mistral3Config) are applied before the
    registry lookup.  The model_type scan below handles any remaining cases
    where Pydantic serialization loses the specific subclass.
    """
    from sendnn_inference.multimodal import MM_HF_CFG_REGISTRY

    utils_cls = MM_HF_CFG_REGISTRY.get(type(hf_config))
    if utils_cls is not None:
        return utils_cls

    # Fallback: scan by model_type string for when hf_config is still a base
    # PretrainedConfig after Pydantic deserialization (e.g. HF-format Mistral3
    # whose class was lost in transit to the encoder subprocess).
    model_type = getattr(hf_config, "model_type", "")
    for cfg_cls, cls in MM_HF_CFG_REGISTRY.items():
        if getattr(cfg_cls, "model_type", None) == model_type:
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
    Model loading happens in ``__init__`` so construction raises on failure.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        from fms.models import get_model

        # Spyre always compiles the LLM decoder in float16 (see SpyreCausalLM.get_dtype()).
        # NNPA may return float32 embeddings even when model weights are float16;
        # cast to float16 before writing to SHM so the decoder sees the compiled dtype.
        self._decoder_dtype = torch.float16

        model_config = vllm_config.model_config

        # Must be called before any NNPA or model operations
        SpyrePlatform.maybe_ensure_sendnn_configured(model_config)

        model_name = model_config.model
        # FMS hf_pretrained + variant resolves from HF cache without a separate
        # download step — the workers already downloaded the weights, so FMS finds
        # them in the local HF snapshot cache.  Use model_path instead when the
        # model is already a local directory (avoids an unnecessary cache lookup).
        is_local = os.path.isdir(model_name)
        fms_kwargs: dict = {"model_path": model_name} if is_local else {"variant": model_name}

        logger.info(
            "encoder_process: loading vision-only model %r "
            "(mm_device=%s, mm_dtype=%s, output_dtype=%s)",
            model_name,
            envs_spyre.SENDNN_INFERENCE_MM_DEVICE,
            envs_spyre.SENDNN_INFERENCE_CPU_MM_DTYPE,
            self._decoder_dtype,
        )
        t0 = time.time()
        self.fms_model = get_model(
            "hf_pretrained",
            vision_only=True,
            # Required for AIU/NNPA: fused QKV is not handled efficiently by
            # NNPA hardware; unfused weights use the optimised NNPA path.
            # Workers also pass fused_weights=False (see spyre.py load_weights).
            fused_weights=False,
            **fms_kwargs,
        )

        # resolve_hf_config normalises format-specific configs (e.g. Mistral-format
        # pixtral → Mistral3Config) so the MM_HF_CFG_REGISTRY lookup in
        # _resolve_mm_utils_cls succeeds directly via class-type match.
        normalized_hf_config = SpyreCausalLM.resolve_hf_config(vllm_config)
        self.mm_utils_cls = _resolve_mm_utils_cls(normalized_hf_config)

        self.fms_model.eval()
        self.mm_device = cast_params_for_spyre(
            self.fms_model,
            self.mm_utils_cls.mm_parameter_prefixes,
            is_fp8_model=False,
        )
        logger.info("encoder_process: mm_utils=%s", self.mm_utils_cls.__name__)
        torch.set_grad_enabled(False)
        logger.info("encoder_process: vision model loaded in %.2fs", time.time() - t0)

    def execute_model(self, request) -> torch.Tensor:
        """Encode a single MMEncodeRequest and return a CPU-contiguous tensor."""
        input_ids = torch.tensor(request.prompt_token_ids, dtype=torch.int64).unsqueeze(0)
        with torch.inference_mode():
            embeds = self.mm_utils_cls.get_maybe_mm_embeddings(
                self.fms_model,
                input_ids,
                request.mm_features,
                is_decode=False,
                mm_device=self.mm_device,
            )
        return embeds.to(dtype=self._decoder_dtype).cpu().contiguous()


# ── Process entry point ───────────────────────────────────────────────────────


def encoder_process_main(
    vllm_config: VllmConfig,
    job_queue,
    result_queue,
    stop_event,
) -> None:
    """Entry point for the vision encoder subprocess.

    Loads the vision-only model, signals READY, then serves execute_model jobs.
    Results are written to POSIX SHM; only ``(req_id, shape, dtype)`` metadata
    is put on the result queue so all TP workers can read the embedding
    independently without a rank-0 tensor broadcast.

    ``stop_event`` is a ``multiprocessing.Event`` set by the executor on
    shutdown.  The job loop polls it via a timeout on ``job_queue.get`` so
    the process exits cleanly on both graceful and abrupt server termination.

    Job loop:
      get(MMEncodeRequest) → execute_model → write SHM → put (req_id, shape, dtype)
    Exits when stop_event is set or None sentinel is received.
    """
    logger.info("encoder_process: starting")

    try:
        runner = VisionEncoderRunner(vllm_config)
    except Exception as exc:
        logger.exception("encoder_process: failed to load vision model: %s", exc)
        result_queue.put(f"ERROR: {exc}")
        return

    result_queue.put("READY")
    logger.info(
        "encoder_process: ready, waiting for jobs (torch_num_threads=%d, OMP_NUM_THREADS=%s)",
        torch.get_num_threads(),
        os.environ.get("OMP_NUM_THREADS", "unset"),
    )

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
        try:
            embeds = runner.execute_model(job)

            # Write embedding to POSIX SHM; close our handle without unlinking
            # so all TP workers can still open it by name.  Rank 0 will unlink
            # the block after all workers have read (via _collect_async_mm_results).
            shm = write_embeddings(embeds, req_id)
            shm.close()

            t_elapsed = time.time() - t0
            result_queue.put((req_id, tuple(embeds.shape), embeds.dtype))
            logger.info("maybe_mm_embedding processing time: %.2fms", t_elapsed * 1000)
        except Exception as exc:
            logger.exception("encoder_process: failed to execute_model '%s': %s", req_id, exc)
            result_queue.put((req_id, None, None))
