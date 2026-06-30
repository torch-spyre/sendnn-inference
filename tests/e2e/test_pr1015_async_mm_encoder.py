"""PR #1015 e2e test — async MM encoder wiring through SpyreMultiprocExecutor.

Boots a real vLLM `LLM` with the async MM encoder enabled and the
`granite-vision-3.2-2b` weights, then drives one MM request through the
full stack:

    request → scheduler emits encode job → SpyreMultiprocExecutor submits
    to encoder subprocess → encoder writes embedding to SHM → workers read
    SHM via collective_rpc → scheduler unblocks prefill → decode completes.

This is the only test that exercises the executor → encoder-process →
worker handshake end-to-end, with all four real processes (engine, two TP
workers, encoder). The unit tests for findings 1-3 stub at the executor
boundary; the wiring (queue creation, READY handshake, scheduler /
executor binding, collective_rpc store_mm_embeddings) only runs here.

Runs on CPU in eager mode. No Spyre hardware needed.
"""

from __future__ import annotations

import sys

import pytest

# Ensure the llava_next mm mapping is imported. FMS serialization
# utilities are patched at import time and the patching is not idempotent —
# the existing tests/e2e/test_spyre_mm.py carries the same note.
import sendnn_inference.multimodal.mm_mappings.llava_next  # noqa: F401

pytestmark = [
    pytest.mark.multimodal,
    pytest.mark.cpu,
    pytest.mark.e2e,
    # vLLM's v1 multiproc engine reliably SIGSEGVs (-11) on macOS during
    # init when tensor_parallel_size > 1. Async MM encoder requires TP > 1
    # (platform.py:266-273), so this test can only run on Linux CI.
    pytest.mark.skipif(
        sys.platform == "darwin",
        reason="vLLM multiproc + TP>1 SIGSEGVs on macOS during engine init; "
        "requires Linux to run.",
    ),
]

GVISION_MODEL = "ibm-granite/granite-vision-3.2-2b"
MAX_TOKENS = 4  # keep CPU work small


@pytest.fixture
def async_encoder_env(monkeypatch):
    """Env for the async MM encoder path.

    `SENDNN_INFERENCE_ASYNC_MM_ENCODER=1` and `tensor_parallel_size > 1`
    are both required for `SpyrePlatform.check_and_update_config` to swap
    in `SpyreMultiprocExecutor` (platform.py:266-273).

    We have to UN-set `VLLM_ENABLE_V1_MULTIPROCESSING=0` from
    `_local_envs_for_test.sh` because TP > 1 needs the multiproc engine to
    actually spawn worker processes — the encoder subprocess is then
    spawned from the executor running in the parent process, which is the
    whole architectural reason this PR exists.
    """
    monkeypatch.setenv("SENDNN_INFERENCE_ASYNC_MM_ENCODER", "1")
    monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager")
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)


@pytest.fixture
def llm(async_encoder_env):
    """Real vLLM LLM with TP=2, eager backend, async MM encoder enabled.

    Slow to construct (loads ~4 GB of weights on CPU + spawns 2 workers +
    1 encoder subprocess), so we scope as `function` rather than `module`
    to ensure each test gets a clean encoder subprocess — the executor
    state machine (`_mm_encoder_proc`, `_mm_in_flight`) is fragile across
    abort/retry test scenarios.

    Reuse across tests in this module is OK; use `module` scope if the
    runtime starts to bite.
    """
    from vllm import LLM

    return LLM(
        model=GVISION_MODEL,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_num_seqs=2,
        # Granite-vision's image-token expansion can produce ~1500 tokens
        # for a typical image; 4k gives headroom for one image + chat
        # template + a few hundred response tokens.
        max_model_len=4096,
        # Disable prefix caching so each request goes through the full
        # encode-then-prefill path even if we send the same image twice.
        enable_prefix_caching=False,
    )


def _build_mm_prompt(llm) -> dict:
    """Build a single MM prompt + tiny image suitable for granite vision.

    Reuses the existing `get_single_image_prompts` helper for the chat
    template (it is granite-vision-specific). We only need one prompt for
    these tests.
    """
    from spyre_util import get_single_image_prompts
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(GVISION_MODEL)
    hf_config = AutoConfig.from_pretrained(GVISION_MODEL)
    image_token = processor.decode(hf_config.image_token_index)

    # tile_size matches the vision encoder's expected input.
    [prompt] = get_single_image_prompts(
        num_prompts=1,
        image_token=image_token,
        tile_size=hf_config.vision_config.image_size,
    )
    return prompt


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

    We deliberately don't assert on output text — granite-vision-3.2-2b
    is stochastic enough on CPU eager that exact-match would be fragile
    and bring no value over the boolean "did it complete".
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    [output] = llm.generate([_build_mm_prompt(llm)], sampling_params)

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
