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

import math
import os
import platform
import queue as queue_mod
import time

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

import sendnn_inference.envs as envs_spyre
from sendnn_inference.model_executor.model_loader.spyre import SpyreCausalLM, cast_params_for_spyre
from sendnn_inference.platform import SpyrePlatform, THREADING_ENVS
from sendnn_inference.v1.worker.mm_encoder_cache import MMEncoderCache, cacheable_identifiers
from sendnn_inference.v1.worker.mm_shared_memory import write_embeddings

logger = init_logger(__name__)


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
        # Cross-request cache of pure image features, keyed by mm_hash. A repeat
        # image reuses its features and skips the vision tower; text embedding and
        # merge still run per request so the emitted (merged) embedding is correct.
        # Process-local — there is a single encoder subprocess, so no coordination.
        self.mm_encoder_cache = MMEncoderCache(
            capacity_bytes=envs_spyre.SENDNN_INFERENCE_MM_ENCODER_CACHE_MB * 1024 * 1024
        )
        logger.info("encoder_process: mm_utils=%s", self.mm_utils_cls.__name__)
        torch.set_grad_enabled(False)
        logger.info("encoder_process: vision model loaded in %.2fs", time.time() - t0)

    def execute_model(self, request) -> torch.Tensor:
        """Encode a single MMEncodeRequest and return a CPU-contiguous tensor.

        Three stages (mirroring FMS's ``prepare_inputs_for_generation``): reuse or
        compute the image features (cached by mm_hash), look up the text embeddings,
        and merge. Only the vision tower is skipped on a cache hit; the emitted
        tensor is the full merged embedding, so the downstream SHM/worker path is
        unchanged.
        """
        input_ids = torch.tensor(request.prompt_token_ids, dtype=torch.int64).unsqueeze(0)
        with torch.inference_mode():
            image_features = self._get_or_encode_image_features(request)
            text_embeds = self.mm_utils_cls.embed_text(self.fms_model, input_ids)
            embeds = self.mm_utils_cls.merge_embeddings(
                self.fms_model, input_ids, text_embeds, image_features
            )
        return embeds.to(dtype=self._decoder_dtype).cpu().contiguous()

    def _get_or_encode_image_features(self, request) -> torch.Tensor:
        """Return packed image features for the request, using the mm_hash cache.

        One image per request (matching the mm_mapping asserts). Warmup identifiers
        are never cached (handled by ``MMEncoderCache.is_cacheable``).
        """
        mm_features = request.mm_features
        identifiers = cacheable_identifiers(mm_features)
        cacheable = self.mm_encoder_cache.enabled and len(identifiers) == 1 == len(mm_features)
        identifier = identifiers[0] if cacheable else None

        if cacheable:
            # The mm_hash identifier (sha256 of the image) uniquely identifies the
            # image, so a stored entry is exactly what encode_images would produce.
            cached = self.mm_encoder_cache.get(identifier)  # ty: ignore[invalid-argument-type]
            if cached is not None:
                self.mm_encoder_cache.record_lookup(hit=True)
                logger.info(
                    "encoder_process: vision-encoder-cache hit rate: %.1f%% "
                    "(req '%s', HIT — skipped tower)",
                    self._hit_rate() * 100,
                    request.request_id,
                )
                return cached

        image_embed = self.mm_utils_cls.encode_images(self.fms_model, mm_features, self.mm_device)
        if cacheable:
            self.mm_encoder_cache.put(identifier, image_embed)  # ty: ignore[invalid-argument-type]
            self.mm_encoder_cache.record_lookup(hit=False)
            logger.info(
                "encoder_process: vision-encoder-cache hit rate: %.1f%% (req '%s', MISS)",
                self._hit_rate() * 100,
                request.request_id,
            )
        return image_embed

    def _hit_rate(self) -> float:
        cache = self.mm_encoder_cache
        total = cache.hits + cache.misses
        return (cache.hits / total) if total else 0.0


# ── Process entry point ───────────────────────────────────────────────────────


def _configure_encoder_threads() -> None:
    """Give the encoder subprocess the full cpu_count thread budget.

    Workers each receive ``cpu_count / num_workers`` threads (set by
    SpyrePlatform.configure_thread_settings).  The encoder subprocess inherits
    that reduced count, but vision encoding is a serial CPU/NNPA workload that
    benefits from all available cores.  We restore the full thread count here
    so OMP, DT_PARALLEL_THREADS, torch inter-op, and intra-op thread pools all
    see the right value when NNPA is initialised.

    Worker thread count is reduced proportionally so the total thread budget
    across all processes stays at cpu_count:
        workers: cpu_count / num_workers  (unchanged — set by platform.py)
        encoder: cpu_count
    """
    cpu_count, _ = SpyrePlatform.get_cpu_count()

    if cpu_count is None:
        encoder_cpu_count = None
    elif platform.machine() == "ppc64le":
        cpu_count, _ = SpyrePlatform.get_cpu_count(use_logical_cpus=True)
        if cpu_count is None:
            encoder_cpu_count = None
        else:
            encoder_cpu_count = min(math.ceil(cpu_count), 36)
    else:
        encoder_cpu_count = math.ceil(cpu_count)

    if encoder_cpu_count is None:
        logger.warning(
            "encoder_process: could not determine encoder_cpu_count; "
            "thread configuration unchanged (inherited from parent)"
        )
        return

    # The encoder gets the full cpu_count.
    encoder_threads = max(1, math.ceil(encoder_cpu_count))

    for env in THREADING_ENVS:
        os.environ[env] = str(encoder_threads)

    # torch intra-op thread pool — can be changed at any time.
    torch.set_num_threads(encoder_threads)
    # torch inter-op thread pool — can only be set before parallel work starts;
    # ignore if it's too late (e.g. in unit tests where torch is already warm).
    try:
        torch.set_num_interop_threads(encoder_threads)
    except RuntimeError as e:
        logger.warning(
            "encoder_process: could not set inter-op threads to %d: %s",
            encoder_threads,
            e,
        )

    logger.info(
        "encoder_process: thread config — encoder=%d,(encoder_cpu_count=%.1f)",
        encoder_threads,
        encoder_cpu_count,
    )


def encoder_process_main(
    vllm_config: VllmConfig,
    job_queue,
    result_queue,
    stop_event,
    cancel_queue=None,
) -> None:
    """Entry point for the vision encoder subprocess.

    Loads the vision-only model, signals READY, then serves execute_model jobs.
    Results are written to POSIX SHM; only ``(req_id, shape, dtype)`` metadata
    is put on the result queue so all TP workers can read the embedding
    independently without a rank-0 tensor broadcast.

    ``stop_event`` is a ``multiprocessing.Event`` set by the executor on
    shutdown.  The job loop polls it via a timeout on ``job_queue.get`` so
    the process exits cleanly on both graceful and abrupt server termination.

    ``cancel_queue`` carries req_id strings for aborted requests.  The encoder
    drains it after dequeuing each job so cancelled jobs are skipped before the
    expensive vision-tower forward pass begins.

    Job loop:
      get(MMEncodeRequest) → drain cancel_queue → skip if cancelled
      → execute_model → write SHM → put (req_id, shape, dtype)
    Exits when stop_event is set or None sentinel is received.
    """
    logger.info("encoder_process: starting")

    # ── Thread configuration ──────────────────────────────────────────────────
    # The encoder subprocess inherits the per-worker thread count that the parent
    # set (cpu_count / num_workers).  We override it here to use the full cpu_count
    # so the vision encoder gets maximum CPU/NNPA parallelism.
    #
    # This MUST happen before maybe_ensure_sendnn_configured() because the NNPA
    # backend captures thread-pool settings at import time.
    _configure_encoder_threads()

    try:
        runner = VisionEncoderRunner(vllm_config)
    except Exception as exc:
        logger.exception("encoder_process: failed to load vision model: %s", exc)
        result_queue.put(f"ERROR: {exc}")
        return

    result_queue.put("READY")
    logger.info(
        "encoder_process: ready, waiting for jobs "
        "(torch_num_threads=%d, OMP_NUM_THREADS=%s, DT_PARALLEL_THREADS=%s)",
        torch.get_num_threads(),
        os.environ.get("OMP_NUM_THREADS", "unset"),
        os.environ.get("DT_PARALLEL_THREADS", "unset"),
    )

    # skip_ids: req_ids drained from cancel_queue that have not yet been dequeued
    #   as a job — the encode will be skipped when the job arrives.
    # processed_ids: tombstones for completed encodes — if a cancel arrives after
    #   the encode is done, discard the tombstone instead of adding to skip_ids.
    skip_ids: set[str] = set()
    processed_ids: set[str] = set()

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

        # Drain the cancel queue before processing this job so that any
        # cancellations that arrived while we were waiting are captured.
        if cancel_queue is not None:
            while True:
                try:
                    rid = cancel_queue.get_nowait()
                    if rid in processed_ids:
                        processed_ids.discard(rid)
                        logger.debug(
                            "encoder_process: late cancel for req '%s' (already done)", rid
                        )
                    else:
                        skip_ids.add(rid)
                        logger.debug("encoder_process: pre-cancel for req '%s'", rid)
                except queue_mod.Empty:
                    break

        req_id = job.request_id

        # Skip encode for cancelled requests; send abort result so the scheduler
        # can clean up promptly instead of waiting for a timeout.
        if req_id in skip_ids:
            skip_ids.discard(req_id)
            # Carry the current cumulative cache counters even on skip so the
            # scheduler always sees the latest hit/miss totals.
            cache = runner.mm_encoder_cache
            result_queue.put((req_id, None, None, cache.hits, cache.misses))
            logger.debug("encoder_process: skipped encode for cancelled req '%s'", req_id)
            continue

        t0 = time.time()
        try:
            embeds = runner.execute_model(job)

            # Write embedding to POSIX SHM; close our handle without unlinking
            # so all TP workers can still open it by name.  Rank 0 will unlink
            # the block after all workers have read (via _collect_async_mm_results).
            shm = write_embeddings(embeds, req_id)
            shm.close()

            t_elapsed = time.time() - t0
            # Cumulative encoder-cache counters ride along so the scheduler can
            # surface the real MM cache hit rate (see scheduler.make_stats).
            cache = runner.mm_encoder_cache
            result_queue.put((req_id, tuple(embeds.shape), embeds.dtype, cache.hits, cache.misses))
            # Tombstone: a late cancel may still arrive on cancel_queue for this req_id.
            processed_ids.add(req_id)
            logger.info("maybe_mm_embedding processing time: %.2fms", t_elapsed * 1000)
        except Exception as exc:
            logger.exception("encoder_process: failed to execute_model '%s': %s", req_id, exc)
            cache = runner.mm_encoder_cache
            result_queue.put((req_id, None, None, cache.hits, cache.misses))
            processed_ids.add(req_id)
