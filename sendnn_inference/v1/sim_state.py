"""Process-local virtual-clock state for sim mode.

The runner advances the clock once per forward step (charging prefill_ms or
decode_ms) and per-request bookkeeping accumulates in `_records`. When a
request finishes, the runner calls `finalize_and_write` which appends a
JSONL line of virtual stats to `<perf_dir>/sim_metrics.jsonl`.

A separate file (rather than substituting into vLLM's request_metrics.jsonl)
avoids the AsyncLLM process boundary: the FileStatLogger that emits
request_metrics.jsonl runs in a different process from the runner, so it
cannot see SimState. Sim mode disables that logger so only sim_metrics.jsonl
is written.

Token timestamps and ITL: each forward step advances the global virtual
clock. We record, per request, the end-time of every prefill step and every
decode step it participates in. The first sampled token is produced by the
*last* prefill chunk; subsequent tokens come from each decode step. This
gives a per-token virtual timeline and meaningful ITL — the gap between
two consecutive decode tokens widens whenever an intervening prefill of
another request happens.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock

from vllm.v1.core.sched.output import SchedulerOutput

import sendnn_inference.envs as envs_spyre


@dataclass
class _RequestSimRecord:
    virtual_arrival: float
    last_prefill_end: float | None = None
    decode_step_ends: list[float] = field(default_factory=list)
    virtual_completion: float | None = None
    num_prefill_chunks: int = 0


class SimState:
    def __init__(self) -> None:
        self.virtual_clock_seconds: float = 0.0
        self._records: dict[str, _RequestSimRecord] = {}
        self._lock = Lock()
        self._fp = None

    def _ensure_file(self):
        if self._fp is not None:
            return
        out_dir = Path(envs_spyre.SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sim_metrics.jsonl"
        if path.exists():
            path.unlink()
        self._fp = path.open("a", buffering=1)

    def has_record(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self._records

    def record_step(
        self,
        is_prompt: bool,
        prefill_ms: float,
        decode_ms: float,
        scheduler_output: SchedulerOutput,
    ) -> None:
        step_seconds = (prefill_ms if is_prompt else decode_ms) / 1000.0
        end_t = self.virtual_clock_seconds + step_seconds
        new_req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
        cached_req_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)

        with self._lock:
            for rid in new_req_ids:
                if rid not in self._records:
                    self._records[rid] = _RequestSimRecord(
                        virtual_arrival=self.virtual_clock_seconds
                    )

            for rid in new_req_ids + cached_req_ids:
                rec = self._records.get(rid)
                if rec is None:
                    rec = _RequestSimRecord(virtual_arrival=self.virtual_clock_seconds)
                    self._records[rid] = rec
                if is_prompt:
                    rec.num_prefill_chunks += 1
                    rec.last_prefill_end = end_t
                else:
                    rec.decode_step_ends.append(end_t)
                rec.virtual_completion = end_t

            self.virtual_clock_seconds = end_t

    def finalize_and_write(
        self,
        req_id: str,
        num_prompt_tokens: int,
    ) -> None:
        prefill_ms = envs_spyre.SENDNN_INFERENCE_SIM_PREFILL_MS
        with self._lock:
            rec = self._records.pop(req_id, None)
        if rec is None:
            return

        # Token emit times (absolute virtual seconds): the first comes from
        # the last prefill chunk; each subsequent from a decode step.
        token_emit_times: list[float] = []
        if rec.last_prefill_end is not None:
            token_emit_times.append(rec.last_prefill_end)
        token_emit_times.extend(rec.decode_step_ends)
        num_generation_tokens = len(token_emit_times)

        if num_generation_tokens == 0:
            # Request never produced a token (e.g., immediate cancel). Skip.
            return

        first_token_t = token_emit_times[0]
        last_token_t = token_emit_times[-1]
        ttft = first_token_t - rec.virtual_arrival
        decode_time = last_token_t - first_token_t  # bench convention
        prefill_time = rec.num_prefill_chunks * prefill_ms / 1000.0

        # ITLs between successive emitted tokens (size = num_generation_tokens - 1)
        itls = [
            token_emit_times[i] - token_emit_times[i - 1]
            for i in range(1, num_generation_tokens)
        ]

        completion = (
            rec.virtual_completion if rec.virtual_completion is not None else last_token_t
        )
        e2e_latency = completion - rec.virtual_arrival
        # In sim mode the scheduler picks a request immediately when it arrives,
        # so there is no front-of-queue wait; report 0 for bench parity.
        queued_time = 0.0
        # Inference time: bench defines it as last_token_ts - scheduled_ts.
        # We approximate scheduled_ts as virtual_arrival.
        inference_time = last_token_t - rec.virtual_arrival
        mean_tpot = decode_time / max(num_generation_tokens - 1, 1)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "request_id": req_id,
            "num_prompt_tokens": num_prompt_tokens,
            "num_generation_tokens": num_generation_tokens,
            "num_prefill_chunks": rec.num_prefill_chunks,
            "num_decode_steps": len(rec.decode_step_ends),
            "virtual_arrival_seconds": rec.virtual_arrival,
            "virtual_completion_seconds": completion,
            "e2e_latency_seconds": e2e_latency,
            "queued_time_seconds": queued_time,
            "prefill_time_seconds": prefill_time,
            "inference_time_seconds": inference_time,
            "decode_time_seconds": decode_time,
            "time_to_first_token_seconds": ttft,
            "mean_time_per_output_token_seconds": mean_tpot,
            "inter_token_latencies_seconds": itls,
        }
        self._ensure_file()
        assert self._fp is not None
        self._fp.write(json.dumps(record) + "\n")


_sim_state: SimState | None = None


def get_sim_state() -> SimState:
    global _sim_state
    if _sim_state is None:
        _sim_state = SimState()
    return _sim_state
