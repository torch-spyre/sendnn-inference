# SPDX-License-Identifier: Apache-2.0
"""Custom request function that captures Spyre per-request metrics injected
into the final SSE usage chunk by the server-side serving patch."""

import json
import sys
import traceback
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
from tqdm import tqdm

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    StreamedResponseHandler,
    _get_chat_content,
    _get_headers,
    _update_headers_common,
    _update_payload_common,
    _validate_api_url,
)


@dataclass
class SpyreRequestFuncOutput(RequestFuncOutput):
    """Extends RequestFuncOutput with Spyre-specific per-request metrics."""

    custom_metrics_dict: dict[str, Any] = field(default_factory=dict)


async def async_request_spyre_chat(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> SpyreRequestFuncOutput:
    """Chat completions request function that additionally parses the
    ``spyre_metrics`` field injected into the final SSE usage chunk."""

    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": (
            request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model
        ),
        "messages": [
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = _get_headers("application/json")
    _update_headers_common(headers, request_func_input)

    output = SpyreRequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content_delta = choices[0]["delta"].get("content")
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                generated_text += content_delta or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                                if (pt := usage.get("prompt_tokens")) is not None:
                                    output.prompt_len = pt
                                # Parse Spyre-specific metrics from the same chunk
                                if spyre_metrics := data.get("spyre_metrics"):
                                    output.custom_metrics_dict = spyre_metrics

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output
