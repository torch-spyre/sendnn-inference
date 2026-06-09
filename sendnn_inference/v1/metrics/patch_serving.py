# SPDX-License-Identifier: Apache-2.0
"""Patch OpenAIServingChat to inject Spyre per-request metrics into the final
SSE usage chunk when SENDNN_INFERENCE_BENCH_METRICS_ENABLED is set.

The final streaming chunk (empty choices, populated usage) is the natural
carrier for per-request metadata because the bench client already parses it.
We intercept only that one chunk per request (one json.loads + json.dumps),
so overhead is negligible.
"""

import dataclasses
import json

from vllm.logger import init_logger

from sendnn_inference.v1.metrics.stats_logger import get_registry

logger = init_logger(__name__)

_patched = False


def patch_serving() -> None:
    """Wrap OpenAIServingChat.chat_completion_stream_generator to inject
    spyre_metrics into the final SSE usage chunk.  Idempotent."""
    global _patched
    if _patched:
        return

    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    except ImportError:
        logger.warning("Could not import OpenAIServingChat — serving patch skipped")
        return

    _original = OpenAIServingChat.chat_completion_stream_generator

    async def _patched_generator(self, request, result_generator, request_id, *args, **kwargs):
        registry = get_registry()
        print(
            f"[SPYRE DEBUG server] _patched_generator called, request_id={request_id}, registry={registry is not None}",
            flush=True,
        )
        async for chunk in _original(self, request, result_generator, request_id, *args, **kwargs):
            if isinstance(chunk, str) and '"usage"' in chunk and '"choices":[]' in chunk:
                print(
                    f"[SPYRE DEBUG server] final usage chunk detected, registry={registry is not None}",
                    flush=True,
                )
                if registry is not None:
                    try:
                        prefix = "data: "
                        data_str = chunk.removeprefix(prefix).rstrip("\n")
                        data = json.loads(data_str)
                        metrics = registry.get_and_clear(request_id)
                        print(
                            f"[SPYRE DEBUG server] registry.get_and_clear({request_id!r}) -> {metrics}",
                            flush=True,
                        )
                        if metrics:
                            data["spyre_metrics"] = dataclasses.asdict(metrics)
                        chunk = f"{prefix}{json.dumps(data)}\n\n"
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        print(f"[SPYRE DEBUG server] exception injecting metrics: {e}", flush=True)
            yield chunk

    OpenAIServingChat.chat_completion_stream_generator = _patched_generator  # ty: ignore[invalid-assignment]
    _patched = True
    print("[SPYRE DEBUG server] patch_serving() applied successfully", flush=True)
    logger.debug("Spyre serving patch applied: spyre_metrics will be injected in final SSE chunk")
