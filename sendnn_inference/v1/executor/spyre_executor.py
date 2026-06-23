"""SpyreMultiprocExecutor — extends vLLM's MultiprocExecutor with an
async MM encoder subprocess.

vLLM spawns worker processes as daemon processes, which means workers
cannot themselves spawn child processes.  By starting the encoder process
here (from the non-daemon executor), we sidestep that restriction.

The executor owns the job/result queues:

- submit jobs from ``_spyre_mm_encode_requests`` in each execute_model call
- collect completed (req_id, shape, dtype) metadata from the result queue
- broadcast metadata to all TP workers via ``collective_rpc("store_mm_embeddings")``
  so each worker reads the embedding from SHM independently — no rank-0
  tensor broadcast needed
- unlink SHM blocks after collective_rpc returns (all workers have read)

This keeps the model runner free of queue concerns: it only maintains
``pending_mm_embeddings`` (keyed by request_id) and reads from it at
prefill time.
"""

import multiprocessing
import multiprocessing.process
import multiprocessing.synchronize
import queue as queue_mod
from typing import Any, Callable

from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from sendnn_inference.v1.worker.mm_shared_memory import cleanup_embeddings_by_name

logger = init_logger(__name__)


class SpyreMultiprocExecutor(MultiprocExecutor):
    """MultiprocExecutor subclass that manages a non-daemon vision encoder
    subprocess for async MM pre-encoding."""

    def _init_executor(self) -> None:
        logger.info("SpyreMultiprocExecutor._init_executor: custom executor active")
        self._mm_encoder_proc: multiprocessing.process.BaseProcess | None = None
        self._mm_job_queue: multiprocessing.Queue | None = None
        self._mm_result_queue: multiprocessing.Queue | None = None
        self._mm_stop_event: multiprocessing.synchronize.Event | None = None
        # Number of encode jobs submitted but not yet collected.
        self._mm_in_flight: int = 0
        super()._init_executor()
        # Do NOT start the encoder process here — warmup runs separately via
        # collective_rpc("compile_or_warm_up_model") and spawning a subprocess
        # while distributed collectives are active corrupts SHM broadcasts.
        # _try_start_mm_encoder is called from collective_rpc below instead.
        # Currently, the warmup step doesn't consume MM encoder process.

    def collective_rpc(  # ty: ignore[invalid-method-override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        **extra_kwargs,
    ) -> Any:
        result = super().collective_rpc(
            method, timeout=timeout, args=args, kwargs=kwargs, **extra_kwargs
        )
        # Start the encoder process after warmup completes — all distributed
        # collectives are done at this point so the subprocess spawn is safe.
        if method == "compile_or_warm_up_model" and self._mm_encoder_proc is None:
            logger.info("SpyreMultiprocExecutor: warmup complete, starting encoder process")
            self._try_start_mm_encoder()
        return result

    def execute_model(self, scheduler_output: Any, non_block: bool = False) -> Any:
        """Submit encode jobs, collect completed results, then run the model step.

        Per-step flow:
          1. Submit new MM encode jobs to the encoder subprocess (non-blocking).
          2. Collect any completed results from the result queue (non-blocking).
          3. For each completed result: tell all TP workers to read from SHM,
             then unlink the SHM block.
          4. Run the normal model step via super().execute_model().
          5. Attach newly_encoded_req_ids to the output for scheduler feedback.
        """
        newly_encoded_req_ids: list[str] = []

        if self._mm_job_queue is not None:
            # Submit new encode jobs.  The scheduler ensures each request
            # appears in _spyre_mm_encode_requests exactly once (tracked via
            # _mm_encoding_submitted), so no dedup is needed here.
            mm_encode_reqs = getattr(scheduler_output, "_spyre_mm_encode_requests", [])
            for req in mm_encode_reqs:
                try:
                    self._mm_job_queue.put_nowait(req)
                    self._mm_in_flight += 1
                    logger.debug("Submitted MM encode job for req '%s'", req.request_id)
                except Exception as exc:
                    logger.debug("MM job queue error for req '%s': %s", req.request_id, exc)

        if self._mm_result_queue is not None and self._mm_in_flight > 0:
            # Collect completed results (non-blocking drain).
            newly_encoded_metadata: list[tuple] = []
            while True:
                try:
                    req_id, shape, dtype = self._mm_result_queue.get_nowait()
                    self._mm_in_flight -= 1
                    if shape is not None and dtype is not None:
                        newly_encoded_metadata.append((req_id, shape, dtype))
                    else:
                        logger.warning(
                            "Encoder process returned error for req '%s' — "
                            "will fall back to inline encoding",
                            req_id,
                        )
                except queue_mod.Empty:
                    break

            if newly_encoded_metadata:
                # Tell all TP workers to read embeddings from SHM independently.
                # collective_rpc is synchronous — when it returns, all workers
                # have finished reading, so the SHM blocks can be safely unlinked.
                self.collective_rpc(
                    "store_mm_embeddings",
                    args=(newly_encoded_metadata,),
                )
                for req_id, _, _ in newly_encoded_metadata:
                    cleanup_embeddings_by_name(req_id)
                newly_encoded_req_ids = [r for r, _, _ in newly_encoded_metadata]
                logger.debug(
                    "Stored async MM embeddings for %d request(s): %s",
                    len(newly_encoded_req_ids),
                    newly_encoded_req_ids,
                )

        # Attach newly_encoded_req_ids to scheduler_output (not to the model
        # runner output) because execute_model(non_block=True) returns a Future,
        # not the resolved ModelRunnerOutput.  The same scheduler_output object
        # is passed to scheduler.update_from_output(), so setting it here means
        # the scheduler will see it regardless of the async/sync execution path.
        if newly_encoded_req_ids:
            scheduler_output._spyre_newly_encoded_req_ids = newly_encoded_req_ids

        # Clear _spyre_mm_encode_requests before dispatching to workers.
        # The async encoder owns all MM encoding jobs
        scheduler_output._spyre_mm_encode_requests = []

        return super().execute_model(scheduler_output, non_block=non_block)

    def _try_start_mm_encoder(self) -> None:
        """Start the vision encoder subprocess.

        The encoder process loads vision weights directly from disk via
        ``get_model(..., vision_only=True)`` — no collective_rpc state-dict
        extraction from rank 0 is required.

        Skipped silently for non-MM models.
        """
        # SpyreMultiprocExecutor is only registered (via platform.py) when
        # SENDNN_INFERENCE_ASYNC_MM_ENCODER=1, so _try_start_mm_encoder is only
        # called for configurations where async encoding is explicitly enabled.
        # No model-type detection needed here — the env var is the gate.
        import sendnn_inference.envs as envs_spyre

        if not envs_spyre.SENDNN_INFERENCE_ASYNC_MM_ENCODER:
            logger.info(
                "SpyreMultiprocExecutor: SENDNN_INFERENCE_ASYNC_MM_ENCODER not set, "
                "skipping encoder process"
            )
            return

        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        try:
            # Plain multiprocessing queues and event,
            # automatically closed when the executor (their owner) exits.
            ctx_q = multiprocessing.get_context("spawn")
            self._mm_job_queue = ctx_q.Queue()
            self._mm_result_queue = ctx_q.Queue()
            self._mm_stop_event = ctx_q.Event()

            # Spawn as daemon so the process is automatically killed when the
            # executor (its parent) exits — including unclean server termination.
            # Using daemon=True is safe here because the encoder process never
            # spawns children of its own.
            # Note: workers are daemon too.
            ctx = multiprocessing.get_context("spawn")
            self._mm_encoder_proc = ctx.Process(
                target=encoder_process_main,
                args=(
                    self.vllm_config,
                    self._mm_job_queue,
                    self._mm_result_queue,
                    self._mm_stop_event,
                ),
                daemon=True,
                name="mm-encoder",
            )
            self._mm_encoder_proc.start()
            logger.info(
                "SpyreMultiprocExecutor: encoder process started (pid=%d), "
                "waiting for vision model to load...",
                self._mm_encoder_proc.pid,
            )

            # Block until the encoder process has finished loading the vision model.
            signal = self._mm_result_queue.get(timeout=300)
            if signal != "READY":
                raise RuntimeError(f"Encoder process startup failed: {signal}")

            logger.info("SpyreMultiprocExecutor: encoder process ready")

        except Exception as exc:
            logger.warning(
                "SpyreMultiprocExecutor: failed to start encoder process (%s: %s); "
                "falling back to Phase 1 blocking encode.",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self._cleanup_encoder()

    def _cleanup_encoder(self) -> None:
        if self._mm_stop_event is not None:
            self._mm_stop_event.set()
        if self._mm_encoder_proc and self._mm_encoder_proc.is_alive():
            self._mm_encoder_proc.join(timeout=5)
            if self._mm_encoder_proc.is_alive():
                self._mm_encoder_proc.terminate()
        self._mm_encoder_proc = None
        self._mm_stop_event = None
        self._mm_job_queue = None
        self._mm_result_queue = None
        self._mm_in_flight = 0

    def shutdown(self) -> None:
        if self._mm_stop_event is not None:
            self._mm_stop_event.set()
        if self._mm_encoder_proc and self._mm_encoder_proc.is_alive():
            self._mm_encoder_proc.join(timeout=10)
            if self._mm_encoder_proc.is_alive():
                self._mm_encoder_proc.terminate()
        super().shutdown()
