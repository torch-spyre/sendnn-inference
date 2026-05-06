# Scheduling and Padding Logic

This page explains how sendnn-inference overrides the vLLM V1 scheduler for decoder (generation) models to respect Spyre hardware constraints, and how requests are padded during prefill and decode execution.

!!! warning
    
    All values and parameters (such as the chunk size) used in the figures are for illustration only; for the real configuration values refer to [configuration](../user_guide/configuration.md)

## Overview

The scheduler uses continuous batching: new requests are prefilled in chunks while previously admitted requests continue decoding. Because Spyre imposes specific constraints on the KV-cache, the scheduler applies additional admission rules on top of vLLM's defaults, and pads requests to meet alignment requirements.

---

## Key concepts

**Block size**: The number of tokens per KV-cache block.

**Chunk size**: The number of tokens processed in a single prefill step. Always a multiple of the block size. A prompt longer than the chunk size is split across multiple successive steps.

**Tkv**: The tkv stands for Token-Key-Value. For a given request, it is the padded token position up to which the KV cache is populated for that request.

**Max-tkv**: The maximum tkv value across all requests currently in the batch.

**Max batch-tkv limit**: A hardware ceiling on the total KV-cache volume. The product `batch_size × max-tkv` must never exceed this limit at any point during execution.

---

## Padding

The padding strategy is designed to meet two constraints specific to Spyre:

- **Decode alignment:** All sequences in the decode batch must occupy the same number of KV-cache blocks at every step. Shorter sequences are padded with full dummy blocks on the left.
- **Prefix caching:** Variable-length left-padding inside a block would corrupt hash-based block comparisons. By aligning prompts to block boundaries (using dummy block ids) we avoid left-padding inside any block, keeping block contents identical for equal token sequences.

### Prefill padding

#### Right-alignment

The last token of a prompt must fall in the last block of its final chunk (i.e., there should be no empty blocks at the right end of the chunk). To achieve this, the prompt is right-padded to the nearest block boundary, and left-padded with enough dummy blocks so that the total padded length is an exact multiple of the chunk size.

Dummy blocks prepended on the left are ignored during attention. Right-padding tokens appended after the last real token are also ignored.

#### Chunked Prefill

For each prefill step, only one chunk is processed. The active chunk is determined by the number of tokens already computed, offset by the left-padding. The last chunked prefill produces one output token.

##### Visualization – Prefill Padding

These visualizations below show the chunked prefill process for a prompt of different lengths.

**Single chunk prefill (prompt len = 15):**

<iframe src="../assets/plots/prefill_single_chunks.html" width="100%" height="450px" frameborder="0"></iframe>

**Three chunks prefill (prompt len = 302):**
<iframe src="../assets/plots/prefill_three_chunks.html" width="100%" height="450px" frameborder="0"></iframe>

### Decode padding

During decode, every request generates exactly one new token per step.

#### Left-padding with full blocks

All requests must share the same number of KV-cache blocks so that the block table is rectangular. The request with the most blocks sets the common width; shorter requests are left-padded with dummy blocks (i.e., any block from the block table). Dummy blocks are masked out by the attention mechanism and do not affect outputs. We always keep the number of left-padding blocks minimal, so when a long request that was forcing other requests to be left-padded finishes, we remove that left-padding for those requests.

#### Right-padding until next block boundary

In addition to full-block left-padding, we also pad the rightmost block of each request (the block containing the tkv) up to its right boundary. These padded tokens are also masked out by the attention mechanism and do not affect outputs.

#### Per-request tkv

Each request's tkv is its left-padding offset plus the number of tokens computed so far, plus one for the token being generated in the current step. The max-tkv for a step is therefore determined by the request with the largest left-padding offset relative to its decode progress.

#### tkv jumping

When a request generates enough tokens to require an additional KV-cache block, it would need one more block than the rest of the batch. To keep all block tables the same width, one fewer dummy block is prepended instead. This causes that request's tkv to jump forward by one block in a single step, while all other requests' tkv values increase by one as usual.

##### Visualization – Decode Padding

The plot below illustrates the full-blocks padding and per-request tkv. We can observe the padding blocks being dynamically appended or removed leading to "jumps" in the tkv values from one step to another when:

1. The tkv value of any of the requests is about to reach a new block (steps 11, 16, 52)
2. A long request finishes, so the other requests can remove their padding blocks (steps 58, 65)

!!! note 
    In the interactive figure below, we don't show the right-padding, because the max-output-tokens is displayed. But it follows the same logic as shown in the [prefill padding visualization](#visualization-prefill-padding): we pad individual tokens until the block's right boundary.

<iframe src="../assets/plots/scheduling_padding_tkv_jump.html" width="100%" height="700px" frameborder="0"></iframe>

---

## Scheduling

### Priority rules

The scheduler enforces a strict priority order:

1. **One prefill at a time.** Only one request can be in its prefill phase at any moment.
2. **Ongoing prefill has priority.** A request that has already started chunked prefill is always scheduled before any new request.
3. **Prefill–decode interleaving.** When interleaving is enabled, two consecutive prefill steps are forbidden if there are any actively decoding requests. This prevents decoding requests from stalling while a long prompt is being prefilled.
    
    !!! note
        Prefill-decode interleaving is enabled by default, but can be disabled or enabled by setting the `SENDNN_INFERENCE_CP_INTERLEAVE_STEPS` environment variable to 0 or 1 respectively.

4. **No idle steps.** If a prefill cannot be scheduled due to constraints, a decode step is run instead — the scheduler never produces an empty output while requests are pending.

### Admission constraints

A request can be admitted for prefill only when all of the following hold.

#### First-chunk constraints

Checked when the request has not yet started prefilling:

- **Batch capacity.** There must be a free slot in the running batch.
- **Single prefill slot.** No new request can start prefilling if one is already in progress.

#### Last-chunk constraints

Checked when the remaining prompt tokens fit within the next chunk — meaning the request is about to complete prefill and join the decode batch:

- **Decode batch capacity.** There must be room in the decode batch for the sequence once it transitions from prefill to decode.
- **Max-model-length constraint.** For every sequence already decoding, and for the new request, the tokens they may still generate must fit within the model's maximum context length.
- **Volumetric constraint.** The product `batch_size × max-tkv` must not exceed the hardware limit at any future decode step. This is verified by the forward-looking check described below.

    ??? info "Volumetric Constraint – Additional Details"
        The hardware imposes a ceiling on the total KV-cache volume: the product of the batch size and max-tkv must not exceed a fixed limit at any step.
    
        The volumetric check answers: *if we admit this request now, will `batch_size × max-tkv` ever exceed the hardware limit?*
    
        The check projects the worst-case future evolution of the batch:
    
        - For the **new request**, its maximum future tkv is its current tkv plus the maximum number of tokens it could still generate.
        - For each **currently decoding request**, its maximum future tkv is its current tkv plus the maximum tokens it could still generate, plus one block to account for a potential padding realignment.  
        
        Because shorter sequences finish earlier and reduce the effective batch size, the constraint is tightest at the steps where the longest-lived requests are still running together. The check iterates over decoding requests in order of increasing maximum future tkv: as each is projected to finish, the batch size shrinks and the binding constraint shifts to the next-longest sequence. The incoming request is accepted only if no projected future state exceeds the hardware limit.
   
##### Visualization – Scheduler Constraints

The visualization below illustrates all the scheduling constraints that can prevent a request from being scheduled at a given step. The different cases mentioned above can be observed in the run.

* **Decode batch capacity:** for both requests 6 and 7, they must wait for a free slot in the decoding batch before they can start prefilling.

* **Prefill-decode interleaving** can be observed during the prefill of requests 1, 2, and 7, where consecutive chunk prefills are separated by an individual decode step. Individual decode steps are also visible between the prefills of consecutive requests.

* **Max-model-length constraint:** the effect can be observed at the time of scheduling request 1. Because the prompt of request 1 is very long, and because the max-output tokens of request 0 is large, scheduling request 1 directly would move the tkv to the fourth block, and the max-output tokens of request 1 would be "pushed" beyond max-context-len. Therefore, after the second chunk completed prefill at step 5, we hold back the third chunk prefill until request 0 completed at step 22.

* **Volumetric constraint:** the prefill of request 4 is deferred due to the volumetric constraint. If it had been scheduled at step 35, the max-tkv would have been 426, leading to a volume of `426 × 4 (requests) = 1704`, which exceeds the maximum accepted volume of 1536. We therefore wait until request 2 finishes at step 45.

<iframe src="../assets/plots/scheduling_admission_constraints.html" width="100%" height="700px" frameborder="0"></iframe>

## Prefix caching

When the KV cache already contains blocks matching the beginning of a prompt, those blocks can be reused without recomputation.

### Skipping full chunks

Whole chunks whose blocks are entirely cached can be skipped. However, the last chunk of a prompt is always recomputed even if all its blocks are cached, because the model must produce the first generated token at the end of the last prefill step.

### Boundary chunk

The chunk that straddles the cache boundary — where some of its blocks are cached and some are not — is always fully recomputed. The KV writes for those blocks are redirected to a dummy block, leaving the cached values untouched while attention still reads from the real cached blocks.

### Scheduling with prefix caching

When prefix caching is enabled, a newly admitted request may have part of its prompt already present in the KV cache. The scheduler accounts for this hit when evaluating first-chunk and last-chunk conditions, so that admission constraints are applied against the effective remaining prompt length rather than the full prompt length.

##### Visualization – Prefix Caching

The visualization below shows a typical cache hit. Since the last prefix block resides in a chunk that needs to be recomputed, it is masked out with a dummy block to avoid duplication of the KV cache.

<iframe src="../assets/plots/prefix_caching_1.html" width="100%" height="700px" frameborder="0"></iframe>


The last chunk must always be recomputed, even if it contains a full prefix hit:

<iframe src="../assets/plots/prefix_caching_2.html" width="100%" height="700px" frameborder="0"></iframe>

Prefix caching can also apply to decoded tokens, as long as they are part of the new prompt:

<iframe src="../assets/plots/prefix_caching_3.html" width="100%" height="700px" frameborder="0"></iframe>