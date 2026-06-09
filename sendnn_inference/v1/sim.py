"""Sim-mode plumbing: no-op model + virtual-clock state.

Activated by SENDNN_INFERENCE_SIM_MODE=1. The runner instantiates
``MockSpyreCausalLM`` instead of ``SpyreCausalLM`` (no FMS load, no
torch.compile, no senlib) and feeds each forward step into ``SimState``,
which advances a virtual clock by ``SIM_PREFILL_MS`` or ``SIM_DECODE_MS``
and accumulates per-request timing. When a request finishes, the runner
calls ``finalize_and_write`` which appends a JSONL line of virtual stats
to ``<perf_dir>/sim_metrics.jsonl``.

A separate output file (rather than substituting into vLLM's
request_metrics.jsonl) avoids the AsyncLLM process boundary: the
FileStatLogger that emits request_metrics.jsonl runs in a different
process from the runner, so it cannot see SimState. Sim mode disables
that logger so only sim_metrics.jsonl is written.

Token timestamps and ITL: each forward step advances the global virtual
clock. We record, per request, the end-time of every prefill step and
every decode step it participates in. The first sampled token is produced
by the *last* prefill chunk; subsequent tokens come from each decode
step. This gives a per-token virtual timeline and a meaningful ITL — the
gap between two consecutive decode tokens widens whenever an intervening
prefill of another request happens.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from transformers import AutoTokenizer

import sendnn_inference.envs as envs_spyre
from sendnn_inference.model_executor.model_loader.spyre import SpyreAttentionMetadata


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockSpyreCausalLM:
    """No-op stand-in for SpyreCausalLM.

    Returns dummy logits without running any real forward pass. Also used
    by unit tests that exercise scheduler/runner logic without a real model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.sampler = Sampler()

        # boolean tensor of length batch size with indices:
        # True for unfinished sequences and
        # False for finished or padded sequences
        self.indices = None

        # number of right pads (relevant for continuous batching only)
        self.n_pads_right = 0

        self.vocab_size = vllm_config.model_config.get_vocab_size()

        # ChunkedPrefillModelRunner.vocab_size reads .fms_model.config.src_vocab_size
        # and .is_multimodal directly; provide minimal shims so warmup works.
        self.is_multimodal = False
        self.fms_model = SimpleNamespace(config=SimpleNamespace(src_vocab_size=self.vocab_size))

        # These variables are here for future test scenarios to use
        self.last_input_ids: torch.Tensor | None = None
        self.last_positions: torch.Tensor | None = None
        self.last_masks: torch.Tensor | None = None
        self.last_is_prompt: bool | None = None
        self.last_attn_metadata: SpyreAttentionMetadata | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, revision=vllm_config.model_config.revision
        )
        self.a_token = self.tokenizer.encode("a", add_special_tokens=False)[0]

    def get_maybe_mm_embeddings(self, *args, **kwargs):
        # This model is not multimodal
        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:
        # These variables are here for future test scenarios to use;
        # NOTE: for now, we always use input IDs since this isn't multimodal.
        self.last_input_ids = input_ids_or_embeds
        self.last_positions = positions
        self.last_masks = masks
        self.last_is_prompt = is_prompt

        forward_context = get_forward_context()

        assert isinstance(forward_context.attn_metadata, SpyreAttentionMetadata)
        self.last_attn_metadata = forward_context.attn_metadata

        batch_size = input_ids_or_embeds.shape[0]

        # make the logits predictable
        logits = torch.zeros(
            (batch_size, self.vocab_size), dtype=torch.float32, device=input_ids_or_embeds.device
        )
        logits[:, self.a_token] = 1
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def set_past_key_value_states(self, num_blocks) -> None:
        pass


# ---------------------------------------------------------------------------
# Virtual-clock state
# ---------------------------------------------------------------------------


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
            token_emit_times[i] - token_emit_times[i - 1] for i in range(1, num_generation_tokens)
        ]

        completion = rec.virtual_completion if rec.virtual_completion is not None else last_token_t
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
