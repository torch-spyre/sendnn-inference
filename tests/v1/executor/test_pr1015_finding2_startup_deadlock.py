"""PR #1015 review — finding 2 (deadlock): encoder startup failure leaves
MM scheduling permanently broken.

`_try_start_mm_encoder` catches *any* exception, logs a warning, calls
`_cleanup_encoder` (which nils every queue handle), and returns. The
comment claims this "falls back to Phase 1 blocking encode" — but there
is no fallback. The scheduler still:

  1. Adds every MM request to `_mm_encoding_submitted` on every step.
  2. Returns False from `can_schedule_prefill` for every MM request
     (because `_mm_encoding_ready` stays empty — nothing populates it).

Result: every MM request hangs forever. The server stays "up" from a
liveness perspective. Text-only requests still work, masking the issue.

Recommended fix: re-raise on startup failure. A loud crash lets the
supervisor restart the process; a silent fallback masks the failure
indefinitely.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


def test_encoder_startup_failure_raises_runtime_error():
    """When the encoder subprocess fails to start (READY handshake returns
    an error sentinel, model load OOMs, get times out, etc.),
    `_try_start_mm_encoder` MUST raise. Today the broad `except Exception`
    swallows the error and returns silently, leaving every subsequent MM
    request to hang on the scheduler gate.
    """
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    from sendnn_inference.v1.executor.spyre_executor import SpyreMultiprocExecutor

    with (
        patch.object(MultiprocExecutor, "_init_executor", return_value=None),
        patch.object(MultiprocExecutor, "execute_model", MagicMock(return_value=MagicMock())),
        patch.object(MultiprocExecutor, "collective_rpc", MagicMock(return_value=None)),
        patch.object(MultiprocExecutor, "shutdown", return_value=None),
    ):
        exc = SpyreMultiprocExecutor.__new__(SpyreMultiprocExecutor)
        exc.vllm_config = MagicMock()
        exc._init_executor()

    # Simulate startup failure: encoder process puts "ERROR: …" on the
    # result queue instead of "READY".
    fake_result_q = MagicMock()
    fake_result_q.get.return_value = "ERROR: vision model load failed"

    fake_ctx = MagicMock()
    fake_ctx.Queue.side_effect = [MagicMock(), fake_result_q]
    fake_ctx.Event.return_value = MagicMock()
    fake_ctx.Process.return_value = MagicMock()

    with (
        patch("sendnn_inference.envs.SENDNN_INFERENCE_ASYNC_MM_ENCODER", True),
        patch("multiprocessing.get_context", return_value=fake_ctx),
        patch("sendnn_inference.v1.worker.mm_encoder_process.encoder_process_main"),
        pytest.raises(RuntimeError),
    ):
        exc._try_start_mm_encoder()
