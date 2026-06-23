# Batched Vision Encoder Pre-Computation

## Background

Multimodal models on Spyre compute the vision encoder on CPU (rank 0 only) and broadcast
embeddings to other ranks via POSIX shared memory. Today this encoding runs serially, once per
request, at the start of that request's first prefill step. MM encoding is expensive operation and in current implementation its blocking, so no other operation like prefill and decode of other requests can run in parallel affecting overall performance.

## Goal

Overlap CPU / NNPA vision encoding with AIU prefill/decode by running the encoder in a separate
subprocess. Embeddings are written to POSIX shared memory and all TP workers read them
independently — no rank-0 broadcast of large tensors. The scheduler gates MM request prefill
on encoding readiness, so a request only enters prefill once its embedding is available.

## Evolution Path

**Phase 1 and 2:** Combined in current implementation. Vision encoding runs
in a dedicated non-daemon subprocess (`mm-encoder`) managed by `SpyreMultiprocExecutor`.
The encoder subprocess loads only the vision model via `get_model(..., vision_only=True)`.
The scheduler submits MM requests for encoding on every step and gates prefill on encoding

**Phase 3 (future):** Enable vision encoder batching within the encoder subprocess. This will further improve the performance by handling all pending MM requests in single batch. This requires
FMS changes to stack same-resolution images instead of
concatenating. See [Phase 3](#phase-3-true-vision-encoder-batching-requires-fms-change) below.

---

## Architecture Diagram

### Current Flow

```
Scheduler picks 1 MM request
  → execute_model():
      encode(request)          # CPU, rank 0, single request
      broadcast embeddings     # SHM + dist.broadcast
      prefill chunk on Spyre
      (repeat for each chunk)
  → decode steps
```

### Implemented Flow

```
Encoder subprocess starts AFTER warmup completes
  (SpyreMultiprocExecutor hooks on collective_rpc("compile_or_warm_up_model"))

Scheduler emits unsubmitted waiting MM requests on EVERY step (prefill and decode).
Scheduler gates MM prefill on _mm_encoding_ready
  (only applies when SENDNN_INFERENCE_ASYNC_MM_ENCODER=1).

SpyreMultiprocExecutor.execute_model() on every step:
  1. Submit new _spyre_mm_encode_requests → job_queue   # non-blocking put_nowait
  2. Drain result_queue (non-blocking)                   # collect completed encodings
     if results:
       collective_rpc("store_mm_embeddings")             # all TP workers read SHM
       cleanup SHM blocks
       set scheduler_output._spyre_newly_encoded_req_ids
  3. super().execute_model() → workers run AIU forward   # concurrent with encoder subprocess
                                                         # encoder process runs in parallel

scheduler.update_from_output():
  _mm_encoding_ready.update(_spyre_newly_encoded_req_ids)

Next schedule() call: request now in _mm_encoding_ready → scheduled for prefill
  add_new_request(): cached_mm_embeddings = pending_mm_embeddings.pop(req_id)
  _prepare_chunked_prefill(): uses cached embeddings, skips inline encoding
```

---

## Changes Summary

| File | Change |
|---|---|
| `sendnn_inference/platform.py` | Register `SpyreMultiprocExecutor` when `SENDNN_INFERENCE_ASYNC_MM_ENCODER=1` and TP > 1 |
| `sendnn_inference/v1/executor/spyre_executor.py` | `SpyreMultiprocExecutor`: override `execute_model` to submit encode jobs, collect results, call `store_mm_embeddings` on workers |
| `sendnn_inference/v1/worker/mm_encoder_process.py` | `VisionEncoderRunner` + `encoder_process_main`: load vision-only model, serve encode jobs, write embeddings to SHM |
| `sendnn_inference/v1/worker/spyre_worker.py` | Add `store_mm_embeddings` — delegates to model runner |
| `sendnn_inference/v1/worker/spyre_model_runner.py` | Add `pending_mm_embeddings` dict, `store_mm_embeddings` (reads from SHM), `_compute_and_cache_mm_embeddings` as inline fallback for warmup; consume in `add_new_request` |
| `sendnn_inference/v1/core/scheduler.py` | Add `MMEncodeRequest` dataclass; emit encode jobs every step; track `_mm_encoding_submitted` / `_mm_encoding_ready`; gate MM prefill on encoding readiness (async mode only); update state in `update_from_output` and `finish_requests` |
| `sendnn_inference/model_executor/model_loader/spyre.py` | Extract `cast_params_for_spyre` as module-level function reusable by encoder subprocess |
| `sendnn_inference/envs.py` | Add `SENDNN_INFERENCE_ASYNC_MM_ENCODER` env var (default 0) |

Non-MM requests, the warmup path, chunked prefill logic, and TP broadcast are unaffected.



## Alternatives considered

### Threading (abandoned)

**What we tried:** Start a `threading.Thread` in the worker model runner. The thread uses the
already-loaded `fms_model` directly (no copy) and encodes waiting MM requests in the
background while the AIU runs.

**Why it failed:** Spyre operations and vision encoding both are blocking operations. The background thread cannot make any progress during AIU execution. Encoding only runs in tiny Python gaps between AIU calls and prefill / decode gets impacted by encoding operations.

**Verdict:** No benefit. Reverted.

### Subprocess from worker (abandoned)

**What we tried:** Start a `multiprocessing.Process` from inside the worker's `load_model` or
`complete_warmup`.

**Why it failed:** vLLM spawns worker processes as **daemon processes**
(`multiprocessing.Process(daemon=True)`). Python forbids daemon processes from spawning
children (`AssertionError: daemonic processes are not allowed to have children`).

**Verdict:** Architecturally impossible from a worker process.

### Subprocess from MultiprocExecutor (**implemented**)

**The idea:** vLLM's `MultiprocExecutor` runs in the **main (non-daemon) process**. Any
process it spawns is also non-daemon. By subclassing `MultiprocExecutor` as
`SpyreMultiprocExecutor`, we can start the encoder process at the executor level.

**Model weight loading:** FMS now supports `get_model(..., vision_only=True)`, which loads
only vision tower + projector + text embedding from the checkpoint, skipping the LLM
decoder. The encoder subprocess calls this directly.

**SHM-based result delivery:** The encoder process writes completed embeddings to POSIX SHM
and puts only `(req_id, shape, dtype)` metadata on the result queue (no large tensors in the
queue). The executor calls `collective_rpc("store_mm_embeddings", metadata)` so all TP workers
read from SHM independently — no rank-0 to others tensor broadcast.

**Scheduler-level encoding readiness gate:** The scheduler tracks `_mm_encoding_submitted` and
`_mm_encoding_ready` sets. MM requests are only eligible for prefill when their encoding is
confirmed complete. Text-only requests are completely unaffected. The scheduler submits encoding
jobs on every step (prefill AND decode) so the encoder stays ahead of the prefill queue.


## Phase 3: Add Vision Encoder Batching

For N same-resolution images, the vision transformer can runs once on `[N, P, D]`
instead of N times on `[1, P, D]`. CPU / NNPA matmul efficiently scales with batch size, so the
single batched call should be significantly faster than N sequential calls — particularly
for large images where the `P²` self-attention dominates.
