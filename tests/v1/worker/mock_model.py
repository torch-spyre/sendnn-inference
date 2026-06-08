from typing import Any
import os

import pytest
import torch
from vllm import EngineArgs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput
from vllm.v1.request import Request
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

from sendnn_inference.model_executor.model_loader.spyre import SpyreAttentionMetadata
from sendnn_inference.platform import SpyrePlatform
from sendnn_inference.v1.worker.spyre_model_runner import (
    ChunkedPrefillModelRunner,
    ChunkedPrefillPlan,
)
from sendnn_inference.v1.core.scheduler import ChunkedPrefillSpyreScheduler

from spyre_util import REFERENCE_MODELS, patch_environment


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


class InstrumentedModelRunner(ChunkedPrefillModelRunner):
    ALL_SLICE = slice(None)
    DEFAULT_TEST_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        vllm_config.cache_config.num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks_override
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        self._model = MockSpyreCausalLM(vllm_config=vllm_config)

        kv_cache_spec = next(iter(self.get_kv_cache_spec().items()))

        group_spec = KVCacheGroupSpec(
            layer_names=[kv_cache_spec[0]], kv_cache_spec=kv_cache_spec[1]
        )
        self.kv_cache_config = KVCacheConfig(
            num_blocks=vllm_config.cache_config.num_gpu_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[group_spec],
        )

        self.scheduler = ChunkedPrefillSpyreScheduler(
            vllm_config=vllm_config,
            kv_cache_config=self.kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            include_finished_set=False,
            log_stats=False,
            block_size=vllm_config.cache_config.block_size,
        )

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        # These variables are here for future test scenarios to use
        self.model.last_input_ids = None
        self.model.last_positions = None
        self.model.last_masks = None
        self.model.last_is_prompt = None
        self.model.last_attn_metadata = None

        return super().execute_model(scheduler_output, **kwargs)

    def execute_new_request(self, request: Request) -> ModelRunnerOutput:
        scheduler_output = self._schedule_new_request(request)
        output = self.execute_model(scheduler_output)
        self.scheduler.update_from_output(scheduler_output, output)
        return output

    def execute_running_requests(
        self,
    ) -> ModelRunnerOutput:
        scheduler_output = self._schedule_running_requests()
        output = self.execute_model(scheduler_output)
        self.scheduler.update_from_output(scheduler_output, output)
        return output

    def _schedule_new_request(self, request: Request) -> SchedulerOutput:
        self.scheduler.add_request(request)
        return self.scheduler.schedule()

    def _schedule_running_requests(
        self,
    ) -> SchedulerOutput:
        return self.scheduler.schedule()

    def _extra_sched_output_kwargs(self) -> dict[str, Any]:
        return {
            "scheduled_spec_decode_tokens": {},
            "scheduled_encoder_inputs": {},
            "num_common_prefix_blocks": [],
            "free_encoder_mm_hashes": [],
        }

    def assert_block_tables_and_slot_mappings(
        self,
        block_tables: list[list[int]],
        # just the first slot index divided by the block_size,
        # will be expanded until the 64th
        slot_mappings: list[list[int]],
        block_size: int = 64,
        slot_slice: slice = ALL_SLICE,
    ) -> None:
        attn_metadata = self.model.last_attn_metadata
        expected_block_table = torch.tensor(block_tables)

        assert torch.equal(attn_metadata.block_table, expected_block_table), (
            f"Actual block table {attn_metadata.block_table}"
        )

        slot_mapping_tensor_list = []
        for slot_mapping in slot_mappings:
            slot_mapping_tensor = torch.arange(block_size, dtype=torch.int64).repeat(
                len(slot_mapping)
            )
            slot_mapping_tensor += (
                torch.tensor(slot_mapping, dtype=torch.int64)
                .repeat_interleave(block_size)
                .mul_(block_size)
            )
            slot_mapping_tensor_list.append(slot_mapping_tensor[slot_slice])
        expected_slot_mapping = torch.stack(slot_mapping_tensor_list)

        assert torch.equal(attn_metadata.slot_mapping, expected_slot_mapping), (
            f"Actual slot mapping {attn_metadata.slot_mapping}"
        )

    def verify_model_runner_output(
        self,
        model_runner_output: ModelRunnerOutput,
        req_ids: list[str],
        num_sampled_token_ids: int,
        tkv: int,
        n_free_blocks: int,
        left_padding: dict[str, int],
        prefix_cache_hit_len: dict[str, int] | None = None,
    ) -> None:
        assert model_runner_output.req_ids == req_ids
        assert len(model_runner_output.sampled_token_ids) == num_sampled_token_ids
        assert model_runner_output.tkv == tkv
        sched_free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        assert sched_free_blocks == n_free_blocks, (
            f"actual free blocks: {sched_free_blocks}, expected: {n_free_blocks}"
        )
        assert model_runner_output.left_padding == left_padding, (
            f"Expected {left_padding}, got {model_runner_output.left_padding}"
        )
        if prefix_cache_hit_len is not None:
            assert model_runner_output.prefix_cache_hit_len == prefix_cache_hit_len

    def verify_chunk_plan(
        self,
        chunk_plan: ChunkedPrefillPlan,
        chunk_count: int,
        padding_blocks: int,
        usable_cache_blocks: int = 0,
        total_cache_blocks: int = 0,
    ) -> None:
        assert chunk_plan.chunk_count == chunk_count
        assert chunk_plan.padding_blocks == padding_blocks
        assert chunk_plan.usable_cache_blocks == usable_cache_blocks
        assert chunk_plan.total_cache_blocks == total_cache_blocks

    @classmethod
    def build(
        cls,
        monkeypatch: pytest.MonkeyPatch,
        enable_prefix_caching: bool = True,
        model_name: str = DEFAULT_TEST_MODEL,
        max_num_seqs: int = 2,
        max_model_len: int = 512,
        max_num_batched_tokens: int = 128,
        available_blocks: int | None = None,
    ) -> ChunkedPrefillModelRunner:
        """A fixture that returns a model runner configured for prefix caching."""

        os.environ.pop("VLLM_DT_MAX_BATCH_TKV_LIMIT", None)

        patch_environment(
            warmup_shapes=None,
            backend="eager",
            monkeypatch=monkeypatch,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        model = REFERENCE_MODELS[model_name]
        engine_args = EngineArgs(
            model=model.name,
            tokenizer=model.name,
            revision=model.revision,
            tokenizer_revision=model.revision,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            num_gpu_blocks_override=available_blocks,
            logits_processors=[],
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )
        vllm_config = engine_args.create_engine_config()

        model_runner = cls(
            vllm_config=vllm_config,
            is_driver_worker=True,
            rank=0,
        )

        model_runner.pre_warmup()
        model_runner.complete_warmup()

        return model_runner
