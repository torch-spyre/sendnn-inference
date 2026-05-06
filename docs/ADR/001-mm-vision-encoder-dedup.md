# ADR-001: De-duplicate Vision Encoder Computation Across TP Workers

## Context

When running multimodal requests with tensor parallel (TP), each TP worker independently calls `get_maybe_mm_embeddings()` in `mistral3.py` during chunked prefill. This function runs on CPU and performs vision encoding followed by language model embedding projection.

For TP=4, this means:
- Vision encoding runs **4×** for every multimodal request (Duplication)
- All workers **block** synchronously while encoding runs  - the causes ITL spike even though MM doesn't get processed during decode.
- There is **no batching** of concurrent multimodal requests through the vision encoder

### Key constraint

`get_maybe_mm_embeddings()` lives in the model code, which is loaded inside worker processes. The engine process and scheduler have no access to the vision encoder. **Encoding must happen inside a worker process.**


## Decision

Encoding is assigned exclusively to **rank-0's worker process**, running in a **background thread pool**. Encoding is triggered as early as possible — the first time rank-0 sees `mm_features` for a request in `execute_model` — so that encoding overlaps with prefill chunk processing rather than blocking it.

The result is written to a `SharedMemory` block. Each rank has a **dedicated result queue** fed by rank-0's background thread. Each worker's **polling thread** watches its queue, reads from shared memory, and sets a `threading.Event` to unblock the main thread when the embedding is ready.

### Why rank-0, not a racing claim among workers

`get_maybe_mm_embeddings` must run exactly once per request. Assigning this responsibility statically to rank-0 eliminates any need for a claim queue, compare-and-swap, or cross-worker racing. Rank-0 always encodes; ranks 1–3 always consume.

### Goals

#### Achieved
- Vision encoding runs **once per request**, regardless of TP degree
- Encoding runs in a **background thread** — rank-0's main thread is not blocked
- Encoding runs off the main thread, so the main thread is not blocked. True overlap with the forward pass requires the scheduler to defer dispatch until the embedding is ready (future work).

#### Pending

- Still no explicit batching currently, which could improve performance further
- Only partially helps with ITL Spike
    - Since this solution reduces duplicate encoding, it will improve encode time, thus help with reducing prefill overbload and blocking
    - Concurent requests still encode sequentially on rank-0's thread pool. Generally prefills are still blocked until we get embeddings for all models.  So if worker is occupied by the prefill step, in-progress decode steps for other requests will be queued behind it, causing delays.

---

## Architecture - For TP=4 case

```
Worker process — rank 0
│
├── main thread (execute_model)
│   │
│   ├── first sight of mm_features?
│   │       YES → submit to thread pool
│   │       NO  → already submitted
│   │
│   ├── do other chunk work...
│   │
│   └── before forward pass:
│           future.result()
│           (instant if done, waits if not)
│
├── encoder thread pool
│   │
│   └── get_maybe_mm_embeddings()
│           → write to SharedMemory
│           → put metadata into
│               result_queue_rank0
│               result_queue_rank1
│               result_queue_rank2
│               result_queue_rank3
│
└── polling thread
        └── result_queue_rank0.get()
                → local_events[req_id].set()


Worker processes — rank 1, 2, 3
│
├── main thread (execute_model)
│   │
│   ├── embedding ready? (event check)
│   │       YES → read SharedMemory, proceed
│   │       NO  → event.wait()
│   │             (waits only for remaining
│   │              encoding time)
│   │
│   └── forward pass with embedding
│
└── polling thread
│
└── result_queue_rankN.get()
    → read SharedMemory
    → local_events[req_id].set()
```