#!/usr/bin/env python3
"""Generate scheduler simulation data for the visualization.

Each request is defined as a tuple:
    (arrival_step, prompt_len, max_output_tokens, total_tokens_generated)

Run:
    python docs/mkdocs/data/generate_example.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    max_model_len: int
    max_num_seqs: int
    block_size: int
    chunk_size: int
    max_batch_tkv_limit: int | None = None
    interleave: bool = False


@dataclass
class RequestDef:
    arrival_step: int
    prompt_len: int
    max_output_tokens: int
    total_tokens_generated: int


@dataclass
class _Waiting:
    id: str
    prompt_len: int
    max_tokens: int


@dataclass
class _Prefilling:
    id: str
    prompt_len: int
    max_tokens: int
    chunks_total: int
    left_padding: int = 0
    right_padding: int = 0
    chunks_done: int = 0

    def chunk_prompt_token_start(self, chunk_size: int) -> int:
        return max(0, self.chunks_done * chunk_size - self.left_padding)

    def chunk_prompt_token_end(self, chunk_size: int) -> int:
        return min((self.chunks_done + 1) * chunk_size - self.left_padding, self.prompt_len)


@dataclass
class _Decoding:
    id: str
    prompt_len: int
    max_tokens: int
    total_generated: int
    decoded: int = 0

    @property
    def actual_tokens(self) -> int:
        return self.prompt_len + self.decoded

    def blocks_needed(self, block_size: int) -> int:
        return math.ceil(self.actual_tokens / block_size)

    def is_done(self) -> bool:
        return self.decoded >= self.total_generated


def _decode_rows(reqs: list[_Decoding], block_size: int) -> list[dict]:
    """Return a snapshot of the current decode state without advancing any counters."""
    if not reqs:
        return []
    max_blocks = max(req.blocks_needed(block_size) for req in reqs)
    rows: list[dict] = []
    for req in reqs:
        left_pad = (max_blocks - req.blocks_needed(block_size)) * block_size
        tkv = left_pad + req.actual_tokens
        rows.append(
            {
                "id": req.id,
                "prompt_len": req.prompt_len,
                "max_tokens": req.max_tokens,
                "tkv": tkv,
                "padding": left_pad,
                "decoded": req.decoded,
                "reserved": req.max_tokens - req.decoded,
            }
        )
    return rows


def _advance_decode(reqs: list[_Decoding], block_size: int) -> tuple[list[dict], list[str]]:
    for req in reqs:
        req.decoded += 1
    rows = _decode_rows(reqs, block_size)
    completed = [req.id for req in reqs if req.is_done()]
    return rows, completed


def _volumetric_ok(
    new_prompt_len: int,
    new_max_tokens: int,
    decoding: list[_Decoding],
    block_size: int,
    limit: int,
) -> bool:
    """Check batch_size × max_tkv <= limit for all projected future states.

    Per-request worst-case max future tkv:
    - New request:           ceil((prompt + max_tokens) / block_size) blocks (no extra)
    - Existing decoding req: max of
        (a) ceil((prompt + max_tokens) / block_size) + 1 blocks
            (for potential future jumps from subsequent admissions)
        (b) (d.prompt_len + d.max_tokens) + worst-case padding jump when the
            new request enters with ceil(new_prompt_len / block_size) blocks,
            using current d.blocks_needed as the minimum (most padding) bound

    Sorted ascending by max future tkv, the constraint at position i is:
        (n - i) × tkv[i] <= limit
    where (n - i) is the number of requests still alive at that projected moment.
    """
    new_blocks_at_join = math.ceil(new_prompt_len / block_size)
    new_max_future_tkv = math.ceil((new_prompt_len + new_max_tokens) / block_size) * block_size
    future_tkvs = [new_max_future_tkv] + [
        max(
            (math.ceil((d.prompt_len + d.max_tokens) / block_size) + 1) * block_size,
            max(0, new_blocks_at_join - d.blocks_needed(block_size)) * block_size
            + d.prompt_len
            + d.max_tokens,
        )
        for d in decoding
    ]
    future_tkvs.sort()
    n = len(future_tkvs)
    return all((n - i) * tkv <= limit for i, tkv in enumerate(future_tkvs))


def simulate(config: Config, request_defs: list[RequestDef]) -> list[dict]:
    arrivals: dict[int, list[tuple[str, RequestDef]]] = {}
    for i, rd in enumerate(request_defs):
        arrivals.setdefault(rd.arrival_step, []).append((str(i), rd))
    req_map: dict[str, RequestDef] = {str(i): rd for i, rd in enumerate(request_defs)}

    waiting: list[_Waiting] = []
    prefilling: _Prefilling | None = None
    decoding: list[_Decoding] = []
    last_was_prefill = False
    steps: list[dict] = []

    for step in range(100_000):
        new_arrivals = arrivals.get(step, [])
        for req_id, rd in new_arrivals:
            waiting.append(_Waiting(req_id, rd.prompt_len, rd.max_output_tokens))

        if step == 0 and new_arrivals:
            steps.append(
                {
                    "step_type": "waiting",
                    "waiting": [
                        {"id": w.id, "prompt_len": w.prompt_len, "max_tokens": w.max_tokens}
                        for w in waiting
                    ],
                    "prefilling": None,
                    "decoding": [],
                }
            )

        if not waiting and prefilling is None and not decoding:
            steps.append(
                {
                    "step_type": "done",
                    "waiting": [],
                    "prefilling": None,
                    "decoding": [],
                }
            )
            break

        completed: list[str] = []

        # Admit the next waiting request into the prefill slot when the slot is
        # free. No volumetric check here — the request can always start prefilling
        # its intermediate chunks; only the last chunk is gated (see below).
        if (
            prefilling is None
            and bool(waiting)
            and len(decoding) < config.max_num_seqs
            and not (config.interleave and last_was_prefill and bool(decoding))
        ):
            w = waiting.pop(0)
            padded_prompt_len = math.ceil(w.prompt_len / config.block_size) * config.block_size
            chunk_count = math.ceil(w.prompt_len / config.chunk_size)
            left_padding = chunk_count * config.chunk_size - padded_prompt_len
            right_padding = padded_prompt_len - w.prompt_len
            prefilling = _Prefilling(
                id=w.id,
                prompt_len=w.prompt_len,
                max_tokens=w.max_tokens,
                chunks_total=chunk_count,
                left_padding=left_padding,
                right_padding=right_padding,
            )

        # Process a prefill chunk only if a request is being prefilled AND either
        # it is not the last chunk yet, OR the volumetric constraint allows it to
        # join the decoding batch. If the last chunk is blocked the request stays
        # in the prefill state and a plain decode step is emitted instead.
        if (
            prefilling is not None
            and not (config.interleave and last_was_prefill and bool(decoding))
            and (
                prefilling.chunks_done + 1 < prefilling.chunks_total
                or config.max_batch_tkv_limit is None
                or _volumetric_ok(
                    prefilling.prompt_len,
                    prefilling.max_tokens,
                    decoding,
                    config.block_size,
                    config.max_batch_tkv_limit,
                )
            )
        ):
            decode_rows = _decode_rows(decoding, config.block_size)
            step_data: dict = {
                "step_type": "prefill",
                "waiting": [
                    {"id": w.id, "prompt_len": w.prompt_len, "max_tokens": w.max_tokens}
                    for w in waiting
                ],
                "prefilling": {
                    "id": prefilling.id,
                    "prompt_len": prefilling.prompt_len,
                    "max_tokens": prefilling.max_tokens,
                    "chunks_total": prefilling.chunks_total,
                    "chunks_done": prefilling.chunks_done,
                    "chunk_prompt_token_start": prefilling.chunk_prompt_token_start(
                        config.chunk_size
                    ),
                    "chunk_prompt_token_end": prefilling.chunk_prompt_token_end(config.chunk_size),
                    "left_padding": prefilling.left_padding,
                    "right_padding": prefilling.right_padding,
                },
                "decoding": decode_rows,
            }
            prefilling.chunks_done += 1
            if prefilling.chunks_done >= prefilling.chunks_total:
                rd = req_map[prefilling.id]
                decoding.append(
                    _Decoding(
                        id=prefilling.id,
                        prompt_len=prefilling.prompt_len,
                        max_tokens=prefilling.max_tokens,
                        total_generated=rd.total_tokens_generated,
                        decoded=1,
                    )
                )
                prefilling = None
            last_was_prefill = True
        else:
            decode_rows, done = _advance_decode(decoding, config.block_size)
            completed.extend(done)
            step_data = {
                "step_type": "decode",
                "waiting": [
                    {"id": w.id, "prompt_len": w.prompt_len, "max_tokens": w.max_tokens}
                    for w in waiting
                ],
                # Non-null when a request is stuck waiting to complete its last
                # prefill chunk (volumetric constraint not yet satisfied).
                # chunk_prompt_token_start/end show the last *completed* chunk (chunks_done-1),
                # not the pending last chunk, so the display reflects what was
                # actually processed so far. Edge case: if no chunk is done yet
                # (chunks_done=0, single-chunk request blocked immediately),
                # fall back to showing the pending chunk 0.
                "prefilling": (
                    {
                        "id": prefilling.id,
                        "prompt_len": prefilling.prompt_len,
                        "max_tokens": prefilling.max_tokens,
                        "chunks_total": prefilling.chunks_total,
                        "chunks_done": prefilling.chunks_done,
                        "chunk_prompt_token_start": max(
                            0,
                            max(0, prefilling.chunks_done - 1) * config.chunk_size
                            - prefilling.left_padding,
                        ),
                        "chunk_prompt_token_end": min(
                            max(1, prefilling.chunks_done) * config.chunk_size
                            - prefilling.left_padding,
                            prefilling.prompt_len,
                        ),
                        "left_padding": prefilling.left_padding,
                        "right_padding": prefilling.right_padding,
                    }
                    if prefilling is not None
                    else None
                ),
                "decoding": decode_rows,
            }
            last_was_prefill = False

        decoding = [d for d in decoding if d.id not in completed]
        if completed:
            step_data["completed"] = completed
        steps.append(step_data)
    else:
        raise RuntimeError("Simulation did not converge in 100,000 steps")

    return steps


def main() -> None:
    config = Config(
        max_model_len=512,
        max_num_seqs=4,
        block_size=64,
        chunk_size=128,
        max_batch_tkv_limit=1024,
        interleave=True,
    )

    # (arrival_step, prompt_len, max_output_tokens, total_tokens_generated)
    requests = [
        # TODO write target setup here
        # RequestDef(0, 15, 3, 3),
        # RequestDef(0, 200, 50, 50),
        # RequestDef(0, 100, 40, 9),
        # RequestDef(0, 80, 30, 30),
        # RequestDef(0, 332, 100, 50),
        # RequestDef(0, 268, 100, 50),
        # RequestDef(66, 145, 100, 50),
        # RequestDef(68, 16, 100, 50),
        # RequestDef(71, 100, 100, 50),
        RequestDef(0, 264, 100, 62),
        RequestDef(0, 45, 100, 71),
        RequestDef(0, 115, 100, 67),
    ]

    config_line = {
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "block_size": config.block_size,
        "chunk_size": config.chunk_size,
        "max_batch_tkv_limit": config.max_batch_tkv_limit,
        "interleave": config.interleave,
    }
    steps = simulate(config, requests)
    lines = [json.dumps(config_line)] + [json.dumps(s) for s in steps]
    output_path = Path(__file__).with_name("simple_example.json")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
