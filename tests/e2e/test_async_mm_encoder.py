"""PR #1015 e2e test — async MM encoder wiring through SpyreMultiprocExecutor.

Boots a real vLLM `LLM` with the async MM encoder enabled and the
nano-gv model (a random-init ~16 MB granite-vision variant built by
`tools/build_nano_gv.py`) then drives one MM request through the full
stack:

    request → scheduler emits encode job → SpyreMultiprocExecutor submits
    to encoder subprocess → encoder writes embedding to SHM → workers read
    SHM via collective_rpc → scheduler unblocks prefill → decode completes.

This is the only test that exercises the executor → encoder-process →
worker handshake end-to-end, with all four real processes (engine, two TP
workers, encoder). The unit tests for findings 1-3 stub at the executor
boundary; the wiring (queue creation, READY handshake, scheduler /
executor binding, collective_rpc store_mm_embeddings) only runs here.

Runs on CPU in eager mode against the nano-gv fixture (see
`REFERENCE_MODELS["joerunde/nano-gv"]` in tests/spyre_util.py), so
warmup takes ~2 s instead of ~2 min. No Spyre hardware needed.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

# Ensure the llava_next mm mapping is imported. FMS serialization
# utilities are patched at import time and the patching is not idempotent —
# the existing tests/e2e/test_spyre_mm.py carries the same note.
import sendnn_inference.multimodal.mm_mappings.llava_next  # noqa: F401
from spyre_util import REFERENCE_MODELS

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu, pytest.mark.e2e]

NANO_GV_MODEL = REFERENCE_MODELS["joerunde/nano-gv"]
MAX_TOKENS = 4  # keep CPU work small
NANO_IMAGE_SIZE = 112  # matches the nano-gv vision_config.image_size


# Env required for the async MM encoder path. See the `llm` fixture body for
# the actual env-setting — it's done there so a module-scoped fixture can use
# it (a function-scoped `monkeypatch` fixture can't be promoted to module).
#
# `SENDNN_INFERENCE_ASYNC_MM_ENCODER=1` + `tensor_parallel_size > 1` are
# required for `SpyrePlatform.check_and_update_config` to swap in
# `SpyreMultiprocExecutor` (platform.py:266-273).
#
# Notable knobs:
#   - VLLM_WORKER_MULTIPROC_METHOD=spawn — vLLM defaults to fork, which
#     SIGSEGVs on macOS when the MM stack pulls in tvm_ffi (via xgrammar).
#   - SENDNN_INFERENCE_TP_MM_SHARING=0 — disables the rank-0 → all-ranks
#     SHM-broadcast embedding share path. Its torch.distributed.broadcast
#     collective hangs during warmup with TP > 1 on macOS (pre-existing
#     bug, not in scope for this test). The async encoder uses a different
#     SHM path (collective_rpc store_mm_embeddings), so disabling sharing
#     here only affects warmup encoding — the async encoder takes over
#     after warmup completes.
#   - VLLM_ENABLE_V1_MULTIPROCESSING=0 from _local_envs_for_test.sh is left
#     intact. It controls the engine-core IPC (in-process vs out-of-process),
#     NOT worker TP — the existing TP=2 tests rely on it staying off.


@pytest.fixture(scope="module")
def llm():
    """Real vLLM LLM with TP=2, eager backend, async MM encoder enabled.

    Module-scoped because:
      1. Startup is the dominant cost (spawn 2 workers + spawn encoder +
         warmup ~= 5 s with nano-gv, but each fresh LLM binds MASTER_PORT
         via torch.distributed rendezvous).
      2. `MASTER_PORT=12345` is fixed in `_local_envs_for_test.sh`, so a
         function-scoped LLM hits `EADDRINUSE` when the second test tries
         to spin up a fresh rendezvous before the OS has released the
         port from the previous test.

    Neither test in this module mutates encoder state in a way that would
    leak into the next.
    """
    # `async_encoder_env` is function-scoped (monkeypatch lifetime). Promote
    # by passing through — pytest accepts a module-scoped fixture depending
    # on a function-scoped one ONLY if the inner one is converted to be
    # module-scoped too. We can't easily do that with monkeypatch, so set
    # env vars directly inside the fixture body instead.
    import os
    os.environ["SENDNN_INFERENCE_ASYNC_MM_ENCODER"] = "1"
    os.environ["SENDNN_INFERENCE_DYNAMO_BACKEND"] = "eager"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["SENDNN_INFERENCE_TP_MM_SHARING"] = "0"
    # Use a distinct MASTER_PORT so back-to-back runs with
    # test_mm_cancel_storm.py don't collide on port 12345 (torch.distributed
    # holds it in TIME_WAIT for ~60s after teardown).
    os.environ["MASTER_PORT"] = "12346"

    from vllm import LLM

    return LLM(
        model=NANO_GV_MODEL.name,
        revision=NANO_GV_MODEL.revision,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_num_seqs=2,
        # nano-gv image_size is 112, which produces ~64 vision patches +
        # chat template overhead — 1024 is more than enough.
        max_model_len=1024,
        # Disable prefix caching so each request goes through the full
        # encode-then-prefill path even if we send the same image twice.
        enable_prefix_caching=False,
    )


def _build_mm_prompt() -> list[dict]:
    """OpenAI-style chat messages with a small synthetic image.

    The image content is meaningless (nano-gv has random weights); we
    just need something the processor can tokenize into an image-token
    span so the async encoder path fires.
    """
    img = Image.new("RGB", (NANO_IMAGE_SIZE, NANO_IMAGE_SIZE), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Happy path: a single MM request completes through the async encoder.
# ---------------------------------------------------------------------------


def test_mm_request_completes_through_async_encoder(llm):
    """Submit one MM request, generate a few tokens, assert completion.

    This is the minimum bar for the PR: the executor spawned an encoder
    subprocess, the encoder loaded vision-only weights, the scheduler
    gated the request on encoding, the executor relayed embeddings via
    SHM + `collective_rpc("store_mm_embeddings")`, and the workers
    consumed `pending_mm_embeddings` during prefill.

    We deliberately don't assert on output text — nano-gv has random
    weights and its outputs are garbage; the boolean "did it complete"
    is what matters for the wiring check.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    output = llm.chat(_build_mm_prompt(), sampling_params)[0]

    # Completion is the assertion. If the async encoder wiring is broken,
    # the request will hang on the scheduler's MM gate (request stays in
    # `_mm_encoding_submitted` forever) and pytest's per-test timeout
    # will fire instead — that is the failure mode this test surfaces.
    assert output.outputs, "no completion returned — request hung on the MM gate?"
    assert output.outputs[0].text or output.outputs[0].token_ids, (
        "request completed but produced no tokens — the prefill path saw an "
        "empty / malformed embedding from SHM"
    )


# ---------------------------------------------------------------------------
# Executor health: encoder subprocess actually started.
# ---------------------------------------------------------------------------


def test_async_encoder_subprocess_is_running(llm):
    """Confirm the executor's encoder subprocess is up after warmup.

    Diagnostic for the wiring path in `SpyreMultiprocExecutor.collective_rpc`
    that starts the encoder on the first `compile_or_warm_up_model` call.
    If this assertion fails the executor was selected but the encoder
    never started — every subsequent MM test in this module will hang on
    the scheduler's gate, so this fails fast with a clear reason.
    """
    from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

    executor = llm.llm_engine.engine_core.engine_core.model_executor  # type: ignore[attr-defined]
    assert isinstance(executor, SpyreMultiprocExecutor), (
        f"expected SpyreMultiprocExecutor, got {type(executor).__name__}; "
        "platform.py did not swap in the async-MM executor — check that "
        "SENDNN_INFERENCE_ASYNC_MM_ENCODER=1 and tensor_parallel_size > 1."
    )
    assert executor._mm_encoder_proc is not None, (
        "executor did not start the encoder subprocess on warmup"
    )
    assert executor._mm_encoder_proc.is_alive(), (
        "encoder subprocess died after startup — check the ERROR sentinel "
        "in the encoder process log"
    )
