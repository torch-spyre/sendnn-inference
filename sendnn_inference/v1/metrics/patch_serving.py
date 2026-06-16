# SPDX-License-Identifier: Apache-2.0
"""Patch OpenAIServingChat and OpenAIServingCompletion to inject Spyre
per-request metrics into the final SSE usage chunk when
SENDNN_INFERENCE_BENCH_METRICS_ENABLED is set.

Metrics are carried from the engine process to the API server process via
RequestOutput.kv_transfer_params["__spyre__"], which already travels over the
ZMQ IPC channel.  The patched generators intercept result_generator to
capture the final RequestOutput, then inject the metrics into the final SSE
usage chunk before yielding it.
"""

import json

from vllm.logger import init_logger

logger = init_logger(__name__)

_patched = False


def _inject_into_chunk(chunk: str, spyre_metrics: dict, request_id: str) -> str:
    """Return chunk with spyre_metrics injected into the usage SSE payload,
    or the original chunk if injection fails."""
    try:
        prefix = "data: "
        data_str = chunk.removeprefix(prefix).rstrip("\n")
        data = json.loads(data_str)
        data["spyre_metrics"] = spyre_metrics
        return f"{prefix}{json.dumps(data)}\n\n"
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to inject spyre_metrics into SSE chunk for %s: %s", request_id, e)
        return chunk


def patch_serving() -> None:
    """Wrap chat and completions stream generators to inject spyre_metrics into
    the final SSE usage chunk.  Idempotent."""
    global _patched
    if _patched:
        return

    _patch_chat()
    _patch_completions()
    _patched = True
    logger.debug("Spyre serving patch applied: spyre_metrics will be injected in final SSE chunk")


def _patch_chat() -> None:
    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    except ImportError:
        logger.warning("Could not import OpenAIServingChat — chat serving patch skipped")
        return

    _original = OpenAIServingChat.chat_completion_stream_generator

    async def _patched_generator(self, request, result_generator, request_id, *args, **kwargs):
        spyre_metrics: dict | None = None

        async def _capturing_generator():
            nonlocal spyre_metrics
            async for res in result_generator:
                if res.finished and res.kv_transfer_params:
                    spyre_metrics = res.kv_transfer_params.get("__spyre__")
                yield res

        async for chunk in _original(
            self, request, _capturing_generator(), request_id, *args, **kwargs
        ):
            if (
                spyre_metrics is not None
                and isinstance(chunk, str)
                and '"usage"' in chunk
                and '"choices":[]' in chunk
            ):
                chunk = _inject_into_chunk(chunk, spyre_metrics, request_id)
            yield chunk

    OpenAIServingChat.chat_completion_stream_generator = _patched_generator  # ty: ignore[invalid-assignment]


def _patch_completions() -> None:
    try:
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    except ImportError:
        logger.warning(
            "Could not import OpenAIServingCompletion — completions serving patch skipped"
        )
        return

    _original = OpenAIServingCompletion.completion_stream_generator

    async def _patched_generator(
        self, request, engine_inputs, result_generator, request_id, *args, **kwargs
    ):
        spyre_metrics: dict | None = None

        async def _capturing_generator():
            nonlocal spyre_metrics
            async for prompt_idx, res in result_generator:
                if res.finished and res.kv_transfer_params:
                    spyre_metrics = res.kv_transfer_params.get("__spyre__")
                yield prompt_idx, res

        async for chunk in _original(
            self, request, engine_inputs, _capturing_generator(), request_id, *args, **kwargs
        ):
            if spyre_metrics is not None and isinstance(chunk, str) and '"usage"' in chunk:
                chunk = _inject_into_chunk(chunk, spyre_metrics, request_id)
            yield chunk

    OpenAIServingCompletion.completion_stream_generator = _patched_generator  # ty: ignore[invalid-assignment]
