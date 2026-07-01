"""PR #1015 e2e cancellation-storm test.

Boots the real OpenAI-compatible `vllm serve` with the async MM encoder
enabled, fires a burst of MM requests, cancels them mid-flight, and then
asserts the server is still responsive.

The failure scenarios this catches:

  - Finding 1: cancelled requests still encode. If the encoder subprocess
    is busy chewing through abandoned large-image jobs, the post-burst
    legitimate request times out and this test fails.

  - Deadlock: if a put_nowait / scheduler-state mismatch leaves the
    encoder or scheduler in a wedged state, /health or the follow-up
    request hangs.

Both are exactly the failure modes the unit tests for findings 1 and 3
sketch in isolation; this test runs the whole stack so we can see what
actually happens under load.

Runs on CPU eager mode against the nano-gv fixture (see
`REFERENCE_MODELS["joerunde/nano-gv"]` in tests/spyre_util.py). No Spyre
hardware needed. Tagged `e2e` because it spawns a full server process
and is slow to start.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time

import httpx
import openai
import pytest
from PIL import Image

# Ensure the llava_next mm mapping is imported. FMS serialization
# utilities are patched at import time and the patching is not idempotent —
# same caveat as tests/e2e/test_spyre_mm.py and test_pr1015_async_mm_encoder.py.
import sendnn_inference.multimodal.mm_mappings.llava_next  # noqa: F401
from spyre_util import REFERENCE_MODELS, RemoteOpenAIServer

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu, pytest.mark.e2e]

NANO_GV_MODEL = REFERENCE_MODELS["joerunde/nano-gv"]
NANO_IMAGE_SIZE = 112  # matches the nano-gv vision_config.image_size

# Tuning knobs — sized to make regressions in the cancel path externally
# visible against a ~6 ms/encode nano-gv model.
#
# `N_CANCELLED` is the storm depth. 100 concurrent cancels is enough to
# thoroughly exercise the scheduler → cancel_queue → encoder drain →
# aborted-result plumbing on every path. Going higher hits diminishing
# returns because the test harness (asyncio + httpx + vLLM request queue)
# becomes the bottleneck long before the encoder does.
#
# `POST_BURST_TIMEOUT` is the ceiling for the follow-up legitimate
# request. With nano-gv it should return in ~100 ms; a completely broken
# cancel path that grinds through all N abandoned encodes serially would
# still fit under a second. 5 s is comfortably above CI noise but
# tight enough that any actual DoS regression fails the test.
N_CANCELLED = 100  # MM requests to fire and immediately cancel
CANCEL_AFTER_SECONDS = 0.5  # how long to let them run before cancelling
POST_BURST_TIMEOUT = 5  # max wait for the fresh request after the storm


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Real `vllm serve` with TP=2, eager backend, async MM encoder on,
    running the nano-gv model.

    Module-scoped because server startup (worker spawn + encoder spawn +
    warmup) is still the dominant cost and both tests in this module can
    share a single server — they don't mutate executor state in ways that
    need a clean restart between cases.

    `RemoteOpenAIServer` takes a `ModelInfo` directly and auto-adds
    `--revision <hash>` to the vllm serve args, so we get the pinned
    revision plumbed through without touching `vllm_serve_args` here.
    """
    # RemoteOpenAIServer merges env_dict over os.environ; we only need to
    # pass the overrides here.
    env_overrides = {
        "SENDNN_INFERENCE_ASYNC_MM_ENCODER": "1",
        "SENDNN_INFERENCE_DYNAMO_BACKEND": "eager",
        # vLLM defaults VLLM_WORKER_MULTIPROC_METHOD=fork, which segfaults
        # on macOS when the MM stack pulls in tvm_ffi (via xgrammar for
        # structured outputs). spawn works on Linux too.
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Disable the synchronous rank-0 → all-ranks SHM-broadcast embedding
        # share path — pre-existing bug that hangs during warmup with
        # TP > 1 on macOS (torch.distributed.broadcast collective doesn't
        # complete). The async encoder uses a different SHM path
        # (collective_rpc store_mm_embeddings), so this only affects the
        # inline warmup encoding.
        "SENDNN_INFERENCE_TP_MM_SHARING": "0",
        # Use a distinct MASTER_PORT so back-to-back runs with
        # test_async_mm_encoder.py don't collide on port 12345 (torch.distributed
        # holds it in TIME_WAIT for ~60s after teardown).
        "MASTER_PORT": "12347",
    }

    with RemoteOpenAIServer(
        NANO_GV_MODEL,
        vllm_serve_args=[
            "--tensor-parallel-size", "2",
            "--enforce-eager",
            "--max-num-seqs", "4",
            # nano-gv image_size is 112, which yields ~64 vision patches
            # + chat template overhead — 1024 is plenty.
            "--max-model-len", "1024",
            "--no-enable-prefix-caching",
            # served-model-name is what clients pass in the OpenAI API's
            # `model` field. Use a short stable alias so tests don't need
            # to know the tmp-path where nano-gv was built.
            "--served-model-name", "nano-gv",
        ],
        env_dict=env_overrides,
        max_wait_seconds=600,
    ) as srv:
        yield srv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_data_url(size_px: int = NANO_IMAGE_SIZE) -> str:
    """Build a base64 data: URL for a synthetic image sized for nano-gv.

    Nano-gv expects 112×112 images (config.vision_config.image_size).
    Image content is meaningless — random weights make output garbage.
    """
    img = Image.new("RGB", (size_px, size_px), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_chat_messages(question: str = "Describe this image.") -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _make_image_data_url()}},
                {"type": "text", "text": question},
            ],
        }
    ]


async def _submit_and_cancel(
    base_url: str, api_key: str, idx: int, cancel_after: float
) -> str:
    """Submit one chat completion, then cancel it after `cancel_after` seconds.

    Uses `httpx.AsyncClient` directly so we can `aclose` the connection
    cleanly mid-stream — this is what propagates the abort to vLLM's
    request handler (vllm hooks the disconnect via the request_id cancel
    channel).

    Returns a short status tag for logging: "cancelled" if we cancelled
    cleanly, "completed_early" if the response beat us to the punch,
    "errored:<msg>" if the server returned an error.
    """
    payload = {
        "model": "nano-gv",
        "messages": _build_chat_messages(f"What is in image #{idx}?"),
        "max_tokens": 64,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
        try:
            # Wrap the request in a task we can cancel.
            task = asyncio.create_task(
                client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            )
            await asyncio.sleep(cancel_after)
            if task.done():
                # The request beat us — server is faster than we thought.
                resp = task.result()
                return f"completed_early:{resp.status_code}"
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return "cancelled"
            except Exception as exc:
                return f"errored:{type(exc).__name__}"
        except Exception as exc:
            return f"errored:{type(exc).__name__}:{exc}"


async def _fire_cancel_storm(base_url: str, api_key: str, n: int) -> list[str]:
    """Fire N requests concurrently and cancel each after CANCEL_AFTER_SECONDS."""
    statuses = await asyncio.gather(
        *(_submit_and_cancel(base_url, api_key, i, CANCEL_AFTER_SECONDS) for i in range(n))
    )
    return list(statuses)


# ---------------------------------------------------------------------------
# Test 1: server stays responsive after a cancellation storm
# ---------------------------------------------------------------------------


def test_server_health_survives_cancel_storm(server: RemoteOpenAIServer):
    """Hammer the server with MM requests, cancel them, then hit /health.

    Even if the encoder is wedged on abandoned jobs, the HTTP server
    thread should remain responsive — this is the cheapest tripwire.
    A non-200 health check post-storm is unambiguous evidence the engine
    deadlocked.
    """
    asyncio.run(
        _fire_cancel_storm(server.url_for("v1"), server.DUMMY_API_KEY, N_CANCELLED)
    )

    # The health endpoint should answer immediately. Give it a tiny grace
    # period in case the in-flight cancellations are still draining.
    deadline = time.time() + 30
    last_status = None
    while time.time() < deadline:
        try:
            r = httpx.get(server.url_for("health"), timeout=5)
            last_status = r.status_code
            if r.status_code == 200:
                return
        except Exception as exc:
            last_status = f"exception: {exc!r}"
        time.sleep(1)

    pytest.fail(
        f"server /health did not return 200 within 30s after a "
        f"{N_CANCELLED}-request cancellation storm (last status: {last_status}). "
        "Symptom of deadlock from finding 2/3 or encoder-pinning DoS from finding 1."
    )


# ---------------------------------------------------------------------------
# Test 2: a fresh MM request completes after the storm
# ---------------------------------------------------------------------------


def test_fresh_request_completes_after_cancel_storm(server: RemoteOpenAIServer):
    """The real bar: after the burst, can a legitimate MM request still
    get through within a reasonable time?

    If the encoder is busy chewing through abandoned jobs (finding 1), this
    request waits behind them and exceeds POST_BURST_TIMEOUT. If the
    scheduler is wedged from put-failure stranding (finding 3), it never
    leaves _mm_encoding_submitted and the request hangs forever.

    Either failure mode shows up as the OpenAI client timing out.
    """
    asyncio.run(
        _fire_cancel_storm(server.url_for("v1"), server.DUMMY_API_KEY, N_CANCELLED)
    )

    client = server.get_client(timeout=POST_BURST_TIMEOUT)
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="nano-gv",
            messages=_build_chat_messages("Describe this image."),
            max_tokens=4,
            temperature=0.0,
        )
    except openai.APITimeoutError:
        elapsed = time.time() - t0
        pytest.fail(
            f"fresh MM request timed out after {elapsed:.1f}s following a "
            f"{N_CANCELLED}-request cancellation storm. Most likely cause: "
            "encoder subprocess is still encoding cancelled requests "
            "(finding 1), or scheduler has a stranded request from a "
            "swallowed put_nowait failure (finding 3)."
        )

    elapsed = time.time() - t0
    assert response.choices, f"empty response after {elapsed:.1f}s"
    assert response.choices[0].message.content is not None, "no content in response"

    # Soft assertion: if the encoder is sluggish but not hung, log it.
    # The hard ceiling is POST_BURST_TIMEOUT (enforced by the OpenAI
    # client timeout); we want a heads-up before we get there.
    if elapsed > POST_BURST_TIMEOUT * 0.7:
        pytest.fail(
            f"fresh MM request took {elapsed:.1f}s after cancellation storm — "
            f"under the {POST_BURST_TIMEOUT}s ceiling but in the danger zone. "
            "Encoder is probably still draining abandoned jobs (finding 1)."
        )
