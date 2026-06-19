# Batched Vision Encoder Pre-Computation

## Background

Multimodal models on Spyre compute the vision encoder on CPU (rank 0 only) and broadcast
embeddings to other ranks via POSIX shared memory. Today this encoding runs serially, once per
request, at the start of that request's first prefill step. Waiting MM requests cannot be encoded
ahead of time, so encoding latency stacks with Spyre prefill latency.

## Goal

Overlap CPU vision encoding with AIU prefill/decode by running the encoder in a separate
subprocess. Embeddings are written to POSIX shared memory and all TP workers read them
independently — no rank-0 broadcast of large tensors. The scheduler gates MM request prefill
on encoding readiness, so a request only enters prefill once its embedding is available.

## Evolution Path

**Phase 1 (baseline, still in use as fallback):** Encoding is blocking, happens at the start
of the `execute_model` call that begins a new prefill. The scheduler emits waiting MM requests
on the first prefill step; the model runner encodes them synchronously before the Spyre forward.
Latency = `encode(waiting) + encode(prefilling) + prefill` all sequential, before the first token.
The Phase 1 path (via `pre_encode_mm_requests`) is still active as a safety fallback when the
encoder subprocess is not running (TP=1, non-MM model, or subprocess startup failure).

**Phase 2 (current implementation on `separate_mm_encoder_scheduling`):** Vision encoding runs
in a dedicated non-daemon subprocess (`mm-encoder`) managed by `SpyreMultiprocExecutor`.
The encoder subprocess loads only the vision model via `get_model(..., vision_only=True)`.
The scheduler submits MM requests for encoding on every step and gates prefill on encoding
readiness. All TP workers read completed embeddings from SHM independently.

**Phase 3 (future):** True vision encoder batching within the encoder subprocess — requires
FMS changes to `PixtralVisionModel.forward` to stack same-resolution images instead of
concatenating. See [Phase 3](#phase-3-true-vision-encoder-batching-requires-fms-change) below.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Encode timing | Blocking, at first-chunk prefill step only | Simpler Phase 1; Phase 2 overlaps via separate process |
| Embedding storage | `pending_mm_embeddings` dict in model runner | Model runner already owns the embedding cache; no upstream changes |
| Batch size cap | `max_num_running_reqs - 1` | Matches max decode batch size; leaves 1 slot for the prefill request |
| Encode scope | All waiting MM requests (any grammar status) | Vision encoding is independent of grammar state |
| Encode during chunked prefill continuation | No — only on first-chunk steps | Simpler; avoids encoding during every step type |

---

## Architecture

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

### Target Flow (Phase 1)

```
Scheduler picks 1 MM request + identifies N waiting MM requests
  → execute_model():
      pre_encode_mm_requests([req2, req3, ...reqN])  # CPU, rank 0, batched (blocking)
      encode(req1) if not already cached             # existing path (no-op if pre-encoded)
      prefill chunk on Spyre
      (repeat for each chunk)
  → decode steps
```

### Target Flow (Phase 2 — future)

```
Persistent encoder process starts alongside the model runner.

Scheduler emits waiting MM requests on EVERY step (prefill and decode),
not just the first prefill step.

execute_model() on any step:
  enqueue([req2, req3, ...]) → encoder process   # non-blocking IPC
  run Spyre AIU forward                           # main process, AIU hardware
                                                  # encoder process runs on CPU in parallel

At the next prefill step (before add_new_request):
  wait for req2 embedding from encoder process    # likely already done; near-zero wait
  consume from pending_mm_embeddings
```

---

## Changes

### 1. New dataclass `MMEncodeRequest`

**Location:** `sendnn_inference/v1/core/scheduler.py` (or a new `sendnn_inference/v1/core/sched/output.py`)

```python
@dataclass
class MMEncodeRequest:
    request_id: str
    prompt_token_ids: list[int]
    mm_features: list  # list[MultiModalFeatureSpec]
```

A lightweight struct that carries the data needed for encoding without passing `Request` objects
across process boundaries.

---

### 2. `ChunkedPrefillSpyreScheduler.schedule()` — populate `mm_encode_requests`

**Location:** `sendnn_inference/v1/core/scheduler.py`

At the point where `new_prefill_candidates` is non-empty (i.e. we are about to start a new
first-chunk prefill), collect waiting MM requests from `holdback_queue` before calling
`super().schedule()`:

```python
if new_prefill_candidates:
    mm_encode_reqs = []
    seen = {r.request_id for r in new_prefill_candidates}
    for req in holdback_queue:
        if req.request_id in seen:
            continue
        if req.mm_features:
            mm_encode_reqs.append(MMEncodeRequest(
                request_id=req.request_id,
                prompt_token_ids=list(req.prompt_token_ids),
                mm_features=req.mm_features,
            ))
            seen.add(req.request_id)
            if len(mm_encode_reqs) >= self.max_num_running_reqs - 1:
                break
```

Attach to output after `super().schedule()` returns (same pattern as `_spyre_grammar_output`):

```python
outputs._spyre_mm_encode_requests = mm_encode_reqs
```

**Key invariant:** The request being prefilled this step is NOT in `mm_encode_requests`. That
request is handled by the existing per-request encoding path in `_prepare_chunked_prefill`.
`mm_encode_requests` only covers *other* waiting requests.

---

### 3. `ChunkedPrefillModelRunner` — `pre_encode_mm_requests()` method

**Location:** `sendnn_inference/v1/worker/spyre_model_runner.py`

Add a dict to the model runner:

```python
self.pending_mm_embeddings: dict[str, torch.Tensor] = {}
```

New method:

```python
def pre_encode_mm_requests(self, encode_requests: list[MMEncodeRequest]) -> None:
    """Batch-encode MM requests that are waiting but not yet prefilling.
    Results are stored in self.pending_mm_embeddings for consumption at prefill time.
    """
    for enc_req in encode_requests:
        if enc_req.request_id in self.pending_mm_embeddings:
            continue  # already encoded from a previous round

        full_tokens = torch.tensor(
            enc_req.prompt_token_ids, dtype=torch.int64, device=self.device
        ).unsqueeze(0)

        if self.rank == 0:
            full_embeds = self.model.get_maybe_mm_embeddings(
                full_tokens, mm_features=enc_req.mm_features, is_decode=False
            )
        else:
            full_embeds = None

        if self.world_size > 1:
            full_embeds = self._broadcast_mm_embeddings(full_embeds)

        if full_embeds is not None:
            self.pending_mm_embeddings[enc_req.request_id] = full_embeds
```

**Refactoring:** Extract the inline SHM broadcast block from `_prepare_chunked_prefill`
(lines 1081–1116) into a `_broadcast_mm_embeddings(embeddings)` helper. This is shared by both
`pre_encode_mm_requests` and the existing single-request path — no behavior change.

---

### 4. `add_new_request()` — consume `pending_mm_embeddings`

**Location:** `sendnn_inference/v1/worker/spyre_model_runner.py`

After constructing `SamplingRequestState`, check the pending dict:

```python
def add_new_request(self, request: NewRequestData):
    ...
    req_state = SamplingRequestState(...)

    # Consume pre-computed embeddings if available
    if request.req_id in self.pending_mm_embeddings:
        req_state.cached_mm_embeddings = self.pending_mm_embeddings.pop(request.req_id)

    self.requests[req_id] = req_state
    ...
```

With `cached_mm_embeddings` already set, `_prepare_chunked_prefill` skips the encoding path via
the existing `if mm_features and request.cached_mm_embeddings is None:` guard and goes directly
to chunk slicing.

---

### 5. `execute_model()` — call `pre_encode_mm_requests` at the right moment

**Location:** `sendnn_inference/v1/worker/spyre_model_runner.py`

```python
def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
    # Batch-encode waiting MM requests before Spyre prefill
    mm_encode_reqs = getattr(scheduler_output, '_spyre_mm_encode_requests', [])
    if mm_encode_reqs:
        self.pre_encode_mm_requests(mm_encode_reqs)

    self.maybe_setup_new_prefill(scheduler_output)
    ...  # rest unchanged
```

`pre_encode_mm_requests` runs before `maybe_setup_new_prefill`, so `pending_mm_embeddings` is
populated before `add_new_request` is called.

---

### 6. Cleanup in `finish_requests()`

**Location:** `sendnn_inference/v1/worker/spyre_model_runner.py`

```python
def finish_requests(self, request_ids, finished_status):
    ...
    if request_ids is not None:
        for rid in request_ids:
            self.pending_mm_embeddings.pop(rid, None)
```

---

## Files Changed

| File | Change |
|---|---|
| `sendnn_inference/v1/core/scheduler.py` | Add `MMEncodeRequest` dataclass; populate `_spyre_mm_encode_requests` in `ChunkedPrefillSpyreScheduler.schedule()` |
| `sendnn_inference/v1/worker/spyre_model_runner.py` | Add `pending_mm_embeddings` dict; extract `_broadcast_mm_embeddings()` helper; add `pre_encode_mm_requests()`; consume pending embeddings in `add_new_request()`; call new method in `execute_model()`; cleanup in `finish_requests()` |

No changes to vLLM core classes, `SpyreModelRunnerOutput`, or the warmup path.

---

## What Phase 1 Does NOT Change

- Spyre still processes exactly one prefill chunk at a time
- The interleaving logic is unchanged
- The tkv / batch constraint checks are unchanged
- Non-MM requests are completely unaffected

---

## Phase 2: Async Encoder Subprocess

### Goal

Overlap CPU vision encoding with AIU prefill/decode so waiting requests (R2, R3, R4) are
pre-encoded while R1 is running on the AIU, eliminating the blocking encode stall before
R1's first token.

---

### Approach 1 — Threading (abandoned)

**What we tried:** Start a `threading.Thread` in the worker model runner. The thread uses the
already-loaded `fms_model` directly (no copy) and encodes waiting MM requests in the
background while the AIU runs.

**Why it failed:** A GIL-check experiment (logging timestamps around the AIU forward) confirmed
that **the Spyre AIU forward never releases the GIL**. The background thread cannot make any
progress during AIU execution. Encoding only runs in tiny Python gaps between AIU calls (~ms),
not the seconds needed for vision encoding.

**Verdict:** No benefit. Reverted.

---

### Approach 2 — Subprocess from worker (abandoned)

**What we tried:** Start a `multiprocessing.Process` from inside the worker's `load_model` or
`complete_warmup`.

**Why it failed:** vLLM spawns worker processes as **daemon processes**
(`multiprocessing.Process(daemon=True)`). Python forbids daemon processes from spawning
children (`AssertionError: daemonic processes are not allowed to have children`).

**Verdict:** Architecturally impossible from a worker process.

---

### Approach 3 — Subprocess from MultiprocExecutor (**implemented**)

**The idea:** vLLM's `MultiprocExecutor` runs in the **main (non-daemon) process**. Any
process it spawns is also non-daemon. By subclassing `MultiprocExecutor` as
`SpyreMultiprocExecutor`, we can start the encoder process at the executor level.

**Model weight loading:** FMS now supports `get_model(..., vision_only=True)`, which loads
only vision tower + projector + text embedding (~4 GB) from the checkpoint, skipping the LLM
decoder. The encoder subprocess calls this directly — no state-dict extraction from rank 0
via `collective_rpc` is needed. This is simpler and more maintainable than the earlier
state-dict sharing approach.

**SHM-based result delivery:** The encoder process writes completed embeddings to POSIX SHM
and puts only `(req_id, shape, dtype)` metadata on the result queue (no large tensors in the
queue). The executor calls `collective_rpc("store_mm_embeddings", metadata)` so all TP workers
read from SHM independently — no rank-0 to others tensor broadcast.

**Scheduler-level encoding readiness gate:** The scheduler tracks `_mm_encoding_submitted` and
`_mm_encoding_ready` sets. MM requests are only eligible for prefill when their encoding is
confirmed complete. Text-only requests are completely unaffected. The scheduler submits encoding
jobs on every step (prefill AND decode) so the encoder stays ahead of the prefill queue.

#### Implementation files

| File | Purpose |
|---|---|
| `sendnn_inference/v1/executor/spyre_executor.py` | `SpyreMultiprocExecutor` subclass |
| `sendnn_inference/v1/worker/mm_encoder_process.py` | Encoder process entry point + `Mistral3VisionProxy` |
| `sendnn_inference/v1/worker/spyre_worker.py` | `get_vision_components_for_encoder`, `set_mm_encoder_queues` |
| `sendnn_inference/v1/worker/spyre_model_runner.py` | Phase 2 execute_model logic |
| `sendnn_inference/platform.py` | Register `SpyreMultiprocExecutor` |

#### Issues encountered during implementation

**1. `distributed_executor_backend` Pydantic validation**
Setting the backend to a string class path was silently dropped by Pydantic's Literal
validator. Fix: pass the **class object** (`SpyreMultiprocExecutor`) directly instead of a
string, which matches the `Type["Executor"]` branch in `Executor.get_class`.

**2. `collective_rpc("load_model")` hook never fires**
vLLM v1 workers call `load_model()` **internally** in their own startup loop, not via
`collective_rpc`. The hook on `"load_model"` never triggered. Fix: hook on
`"compile_or_warm_up_model"` instead, which IS dispatched via `collective_rpc` by the engine.

**3. FMS `extend_adapter` error on import**
`import sendnn_inference.multimodal` in the EngineCore process failed with
`KeyError: 'Source hf must already be registered for architecture llava_next'`.
`llava_next.py` calls `serialization.extend_adapter(...)` at **module level**; in the
EngineCore process no FMS models have been loaded so the base adapter isn't registered.
Fix: avoid the import; check multimodal type via `model_type` string
(`hf_config.model_type in {"mistral3", "llava_next", "pixtral"}`).

**4. `hf_config` deserialized as base `PreTrainedConfig`**
`vllm_config.model_config.hf_config` arrives in the EngineCore process as
`PreTrainedConfig` (base class) after Pydantic serialisation strips the specific subclass.
`isinstance(hf_config, Mistral3Config)` always returned False. Fix: check `model_type`
string attribute instead of the class type.

**5. FMS `get_model` fails: "Can't find the requested checkpoint data"**
The encoder process called `get_model(model_path="mistralai/...")` with the raw HF model ID.
FMS's `hf_pretrained` architecture expects a **local directory path**, not an HF repo ID.
Workers resolve this by calling `download_weights_from_hf(...)` first. Fix: replicate this
step in `_load_vision_proxy`. (This approach was later replaced by the state-dict extraction
approach which avoids disk loading entirely.)

**6. `torch.compile` wraps `fms_model` in `_OptimizedModule`**
`fms_model.named_parameters()` returned nothing because `_OptimizedModule._modules` and
`_parameters` are empty — the compiled wrapper doesn't expose the original module's parameters
through the standard `nn.Module` registry. Fix: unwrap via
`underlying = getattr(fms_model, "_orig_mod", fms_model)` before iterating parameters.

**7. Subprocess spawn during warmup corrupts SHM broadcasts**
Starting the encoder process (via `multiprocessing.Process.start()`) triggers a `fork()+exec()`
on Linux. If this happens while `torch.distributed` collectives are active (e.g., during
warmup's `_broadcast_mm_embeddings`), the brief `fork()` duplicates gloo socket file
descriptors, corrupting the collective's socket state. Workers then see `FileNotFoundError`
when trying to read SHM that rank 0 supposedly wrote.

Fix: hook `_try_start_mm_encoder` on `collective_rpc("compile_or_warm_up_model")` completion
(not `_init_executor`), so the spawn happens after all warmup collectives are done.

**8. TP broadcast desync during inference**
`_mm_job_queue` is set only on rank 0's model runner. The Phase 2 block was guarded by
`if self._mm_job_queue is not None:`, so ranks 1–3 took a different code path. When rank 0
called `dist.broadcast(flags)` inside that block, ranks 1–3 skipped it and went to
`pre_encode_mm_requests` — their next `dist.broadcast` was the SHM broadcast sync A.
Rank 0's `flags` broadcast paired with sync A, producing SHM `FileNotFoundError`.

Fix: move the `dist.broadcast(flags)` **outside** the `_mm_job_queue` guard, so ALL ranks
participate in the flags decision broadcast first, then each rank performs its role.

#### Current architecture (branch `separate_mm_encoder_scheduling`)

```
platform.py
  check_and_update_config():
    if MM model + TP > 1:
      parallel_config.distributed_executor_backend = SpyreMultiprocExecutor

SpyreMultiprocExecutor
  collective_rpc("compile_or_warm_up_model") → _try_start_mm_encoder()
    encoder_process_main(vllm_config, job_queue, result_queue) [non-daemon subprocess]
      VisionEncoderRunner.load_model():
        get_model("hf_pretrained", model_path, vision_only=True)  # ~4GB, no LLM
      → result_queue.put("READY")
      loop: job_queue.get() → encode → write SHM → result_queue.put((req_id, shape, dtype))

  execute_model(scheduler_output):
    submit _spyre_mm_encode_requests → job_queue      # non-blocking
    drain result_queue → newly_encoded_metadata       # non-blocking
    collective_rpc("store_mm_embeddings", metadata)   # all TP ranks read SHM independently
    cleanup_embeddings_by_name(req_id) × N            # executor unlinks SHM
    super().execute_model(scheduler_output)
    output._spyre_newly_encoded_req_ids = newly_encoded_req_ids

ChunkedPrefillSpyreScheduler
  _mm_encoding_submitted: set[str]   # dispatched to encoder, not yet confirmed
  _mm_encoding_ready: set[str]       # confirmed, ready for prefill

  schedule() every step:
    emit mm_encode_requests for waiting MM reqs not in submitted/ready
    _mm_encoding_submitted.add(new req_ids)
    can_schedule_prefill(req):
      if req.mm_features and req_id not in _mm_encoding_ready: return False  # gate!

  update_from_output():
    _mm_encoding_ready.update(_spyre_newly_encoded_req_ids)
    _mm_encoding_submitted.discard(...)

SpyreWorker.store_mm_embeddings(results) → model_runner.store_mm_embeddings(results)
ChunkedPrefillModelRunner.store_mm_embeddings(results):
  for req_id, shape, dtype in results:
    self.pending_mm_embeddings[req_id] = read_embeddings(req_id, shape, dtype)
```

#### Remaining open questions

- Does the encoder subprocess CPU compute actually overlap with the AIU forward?
  Verify via `[parallel-check]` encoder START/END timestamps vs AIU forward timestamps.
- For TP=1: the executor is `UniProcExecutor` (not `SpyreMultiprocExecutor`), so there is
  no encoder subprocess and the Phase 1 blocking path (`pre_encode_mm_requests`) is used.
- `LlavaNext` models: the `can_schedule_prefill` MM gate applies universally to all MM
  models that have `mm_features`; no model-specific changes needed.
- `vllm_config` picklability: verify that `VllmConfig` with a `Mistral3Config` subclass
  round-trips through `multiprocessing.get_context("spawn")` correctly.

---

### Expected Benefit (when fully working)

```
Phase 1:  t=0  → block 3×3.9s = 11.7s → R1 inline encode 3.9s → AIU prefill
Phase 2:  t=0  → R1 inline encode 3.9s → AIU prefill
                 (encoder process encodes R2/R3/R4 concurrently)
          t≈13.5s: R2 prefill → instant (pre-encoded) → zero wait
```

R1's first token improves from t ≈ 11.7 + 3.9 + prefill to t ≈ 3.9 + prefill.
All subsequent requests also benefit from zero encoding stall at their prefill steps.

---

## Phase 3: True Vision Encoder Batching (requires FMS change)

### Current Limitation

`get_mm_embeddings_batch` in
`sendnn_inference/multimodal/mm_mappings/mistral3.py` is named "batch" but processes
requests in a **for loop**:

```python
for input_ids_1d, mm_features in zip(batch_input_ids, batch_mm_features):
    embeds, _ = fms_model.prepare_inputs_for_generation(
        iteration=0,
        input_ids=input_ids_1d.unsqueeze(0),   # one request at a time
        kwargs=Mistral3MMUtils._build_fms_kwargs(mm_features, mm_device),
    )
    results.append(embeds)
```

Each call runs the full Pixtral vision encoder for a single `pixel_values=[1, C, H, W]`.
N waiting requests = N sequential encoder passes.

### Root Cause in FMS

In `PixtralVisionModel.forward` (`fms/models/pixtral_vision.py`), after Conv2d extracts
per-image patches, all images' patches are **concatenated into one flat sequence**:

```python
patch_embeds = torch.cat(
    [p.flatten(1).T for p in patch_embeds_list], dim=0
).unsqueeze(0)                       # [1, N*P_total, D]
```

The transformer then runs **global self-attention across all N images' patches together**.
This makes true batching incorrect: patches from different requests attend to each other,
and compute cost is O((N×P)²) rather than O(N×P²).

### Fix

For images with the **same resolution** (e.g., all 2048×2048 in a benchmark), all images
have the same patch count P. Replace `cat → [1, N*P, D]` with `stack → [N, P, D]`; the
transformer then processes each image as an independent batch element with no cross-image
attention:

**`fms/models/pixtral_vision.py` — `PixtralVisionModel.forward`:**
```python
patch_sizes = [p.shape for p in patch_embeds_list]
if len(patch_embeds_list) > 1 and len(set(patch_sizes)) == 1:
    # All same resolution: stack for true per-image batched attention
    patch_embeds = torch.stack([p.flatten(1).T for p in patch_embeds_list], dim=0)
    # [N, P, D]
    patch_embeds = self.ln_pre(patch_embeds)
    position_ids = get_positions_in_meshgrid([patch_embeds_list[0]])  # [1, P, 2]
    position_ids = position_ids.expand(len(patch_embeds_list), -1, -1)  # [N, P, 2]
else:
    # Variable resolution or single image — existing path unchanged
    patch_embeds = torch.cat(
        [p.flatten(1).T for p in patch_embeds_list], dim=0
    ).unsqueeze(0)
    patch_embeds = self.ln_pre(patch_embeds)
    position_ids = get_positions_in_meshgrid(patch_embeds_list)
```

**`fms/models/mistral3.py` — `_get_image_features`:**
```python
# Was: selected_image_feature.squeeze(0)
# squeeze(0) is wrong when output is [N, P, D] and N > 1
selected_image_feature = selected_image_feature.view(
    -1, selected_image_feature.shape[-1]
)  # [N*P, D] regardless of whether output is [1, N*P, D] or [N, P, D]
```

**`sendnn_inference/multimodal/mm_mappings/mistral3.py` — `get_mm_embeddings_batch`:**  
Replace the for loop with a single call that passes all N requests' pixel_values at once:

```python
@staticmethod
def get_mm_embeddings_batch(fms_model, batch_input_ids, batch_mm_features, mm_device):
    # Build batched pixel_values [N, C, H, W] and combined image_sizes
    all_pixel_values, all_image_sizes = [], []
    for mm_features in batch_mm_features:
        pv, sizes = Mistral3MMUtils._extract_pixel_values_and_sizes(mm_features, mm_device)
        all_pixel_values.append(pv.squeeze(0))   # [C, H, W]
        all_image_sizes.extend(sizes)

    stacked_pv = torch.stack(all_pixel_values, dim=0)  # [N, C, H, W]
    # Single vision encoder forward for all N images
    image_features = fms_model._get_image_features(stacked_pv, all_image_sizes)

    # Split back per request and merge with text embeddings
    results = []
    offset = 0
    for input_ids_1d, mm_features in zip(batch_input_ids, batch_mm_features):
        n_img_toks = (input_ids_1d == fms_model.config.image_token_index).sum().item()
        req_features = image_features[offset : offset + n_img_toks]
        embeds = fms_model._merge_multimodal_embeddings(
            input_ids_1d.unsqueeze(0), ..., req_features, ...
        )
        results.append(embeds)
        offset += n_img_toks
    return results
```

### Variable Resolution

For requests with different image sizes, padding is needed:
- Pad each image's patches to `max_patches` with zeros
- Pass `attn_mask=[N, 1, max_P, max_P]` (block-diagonal, True for valid patches)
- FMS `MultiHeadAttention` accepts `**attn_kwargs` including `attn_mask`

Padding overhead is O(N × max_P²) vs O(N × P²), acceptable when images are similar-sized.

### Expected Benefit

For N same-resolution images, the vision transformer forward runs once on `[N, P, D]`
instead of N times on `[1, P, D]`. CPU matmul efficiency scales with batch size, so the
single batched call should be significantly faster than N sequential calls — particularly
for large images where the `P²` self-attention dominates.
