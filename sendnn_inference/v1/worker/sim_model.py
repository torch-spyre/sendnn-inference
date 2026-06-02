"""Sim-mode model: a no-op stand-in for SpyreCausalLM.

Returns dummy logits without running any real forward pass. Used by the
sim-mode benchmarking flow (SENDNN_INFERENCE_SIM_MODE=1) and by unit tests
that exercise scheduler/runner logic without a real model.

Virtual-time accounting lives in sendnn_inference.v1.sim_state; this class
is intentionally minimal.
"""

from types import SimpleNamespace

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from sendnn_inference.model_executor.model_loader.spyre import SpyreAttentionMetadata


class MockSpyreCausalLM:
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

        return torch.empty(
            (batch_size, self.vocab_size), dtype=torch.float32, device=input_ids_or_embeds.device
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def set_past_key_value_states(self, num_blocks) -> None:
        pass
