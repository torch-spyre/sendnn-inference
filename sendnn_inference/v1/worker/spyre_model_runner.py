import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.activations import get_act_fn
from vllm.model_executor.layers.pooler.seqwise.poolers import (
    pooler_for_classify,
    pooler_for_embed,
)
from vllm.sampling_params import SamplingType
from vllm.tasks import SupportedTask
from vllm.utils.platform_utils import is_pin_memory_available

from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput, SamplerOutput
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.request import Request
from vllm.v1.structured_output.utils import (
    apply_grammar_bitmask as vllm_apply_grammar_bitmask,
)

import sendnn_inference.envs as envs_spyre
import sendnn_inference.utils as utils_spyre
from sendnn_inference.model_executor.model_loader.spyre import (
    BACKEND_LIST,
    SpyreAttentionMetadata,
    SpyreCausalLM,
)
from sendnn_inference.perf_metrics import create_perf_metric_logger
from sendnn_inference.platform import SpyrePlatform
from sendnn_inference.utils import exact_div
from sendnn_inference.v1.sample.spyre_logits_processor import build_logitsprocs_for_cb

# yapf conflicts with ruff for this block
# yapf: disable
from sendnn_inference.v1.worker.spyre_input_batch import (
    BaseInputBatch,
    PoolingInputBatch,
    PoolingRequestState,
    RequestStateT,
    SamplingInputBatch,
    SamplingRequestState,
)

# yapf: enable
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.sample.metadata import SamplingMetadata
else:
    SchedulerOutput = None
    NewRequestData = None
    SamplingMetadata = None

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelForwardInputs:
    input_tokens: torch.Tensor | None  # For non multimodal
    input_embeds: torch.Tensor | None  # For multimodal
    input_positions: torch.Tensor
    is_prompt: bool


@dataclass(frozen=True)
class PoolingForwardInputs(ModelForwardInputs):
    input_masks: torch.Tensor
    token_type_ids: torch.Tensor | None


@dataclass(frozen=True)
class SamplingForwardInputs(ModelForwardInputs):
    """
    Used by the SpyreModelRunner.
    """

    current_tkv_mask: torch.Tensor
    left_padded_prompt_mask: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class SpyreModelRunnerOutput(ModelRunnerOutput):
    # Current tkv: this is the maximum padded request length in the batch)
    tkv: int = 0
    # Left padding of each request that was calculated by model runner
    left_padding: dict[str, int] = field(default_factory=dict)
    # In the case of prefix caching, we may have a much larger cached prefix
    # available than the number of scheduled tokens. In that case, the scheduler
    # needs to update its state to reflect the correct number of computed tokens
    prefix_cache_hit_len: dict[str, int] = field(default_factory=dict)


InputBatchT = TypeVar("InputBatchT", bound=BaseInputBatch)
ModelInputsT = TypeVar("ModelInputsT", bound=ModelForwardInputs)


class BaseSpyreModelRunner(ABC, Generic[InputBatchT, RequestStateT, ModelInputsT]):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        self.is_driver_worker = is_driver_worker
        self.rank = rank
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.pad_token_id = 0

        if self.model_config is not None:
            if self.model_config.hf_config is not None:
                self.pad_token_id = getattr(self.model_config.hf_config, "pad_token_id", None) or 0
            if self.model_config.get_sliding_window():
                logger.warning(
                    "Sliding window is not supported on Spyre. "
                    "The model will run without sliding window."
                )

        if vllm_config.device_config is None:
            self.device_config = DeviceConfig()

        assert self.device_config.device is not None, "Torch device is required"
        self.device = torch.device(self.device_config.device)
        self.pin_memory = is_pin_memory_available()

        # Lazy initialization: after load_model.
        self._model: SpyreCausalLM | None = None

        # Flag to be turned off after warmup is complete
        self.warmup_mode = True

        # Batch state
        self.input_batch = self.build_input_batch()

        # Requests
        self.requests: dict[str, RequestStateT] = {}

    @abstractmethod
    def build_input_batch(self) -> InputBatchT:
        raise NotImplementedError

    @property
    def model(self) -> SpyreCausalLM:
        assert self._model is not None, "model accessed before loading"
        return self._model

    @property
    def is_multimodal(self) -> bool:
        """Indicates whether or not a model is multimodal.
        This should not be called until after the model is loaded.
        """
        if not hasattr(self, "model"):
            raise AssertionError("Cannot check if models are multimodal before loading!")
        return bool(getattr(self.model, "is_multimodal", False))

    def get_mm_utils(self):
        """If the [loaded] model is multimodal, grab the instance of
        the mm utils for the corresponding wrapper class.
        """
        if not self.is_multimodal:
            return None
        return self.model.mm_model_utils

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        This method should generate the KVCache spec by parsing the kv cache
        format from each Attention module in the static forward context.

        In vLLM, this static forward context is populated by the base Attention
        class in the modeling code. Every attention layer populates an entry
        for itself in vllm_config.compilation_config.static_forward_context,
        which is a dictionary of layer_name -> layer for every attention layer.
        This allows the model runner to correctly create the kv cache spec for
        each layer.

        The spyre modeling code currently comes from `fms`, and does not
        integrate with vLLM's modeling classes, so we don't have access to any
        model-agnostic metadata about the attention layers. This just returns a
        dummy value for now.
        """
        # We do at least use the real size from the cache config.
        block_size = self.vllm_config.cache_config.block_size

        attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float16,
        )
        return {"foo": attn_spec}

    def complete_warmup(self):
        """Turn off warmup mode once the warmup is complete"""
        self.warmup_mode = False

    def build_attn_metadata(self, model_input: ModelInputsT) -> SpyreAttentionMetadata:
        # TODO: probably sooner we will need a more sophisticated way to switch
        # build attention metadata based on model/attention. But for now, a
        # simple method override is good enough.
        return None  # ty: ignore

    @abstractmethod
    def update_states(self, scheduler_output: SchedulerOutput):
        raise NotImplementedError

    @SpyrePlatform.inference_mode()
    @abstractmethod
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        raise NotImplementedError


class PoolerAdapter(torch.nn.Module):
    def __init__(self, pooler: torch.nn.Module):
        super().__init__()
        self.pooler = pooler

    def forward(
        self,
        hidden_states: Union[torch.Tensor, tuple[torch.Tensor, ...]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        # Because we're using transformers to load the pooler
        # and classifier layers and the assumption there is that
        # we have a right padded batch, we need to split
        # and at the batch dimension.
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.split(hidden_states, pooling_metadata.prompt_lens.tolist())
        return [self.pooler(h.unsqueeze(dim=0)) for h in hidden_states]


def _cls(input: torch.Tensor) -> torch.Tensor:
    return input[:, 0]


class SpyrePoolingModelRunner(
    BaseSpyreModelRunner[PoolingInputBatch, PoolingRequestState, PoolingForwardInputs],
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        vllm_config: VllmConfig = vllm_config
        self.spyre_warmup_shapes = SpyrePlatform.get_warmup_shapes(vllm_config.scheduler_config)
        # position_ids of all the sequences in current batch
        self._position_ids: torch.Tensor = None
        self.use_token_type_ids = False
        assert self.cache_config.block_size == self.model_config.max_model_len, (
            "cache_config.block_size must be set to model_config."
            "max_model_len to disable any paged attention ops in the base "
            "scheduler."
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._model  # ty: ignore[invalid-return-type]

    def build_input_batch(self) -> PoolingInputBatch:
        return PoolingInputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

    def load_model(self) -> None:
        assert len(self.model_config.architectures) == 1
        task = (
            "classify"
            if self.model_config.architectures[0].endswith("ForSequenceClassification")
            else "embed"
        )

        if task == "embed":
            self._model = AutoModel.from_pretrained(self.model_config.model)
        elif task == "classify":
            class_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model
            )
            if hasattr(class_model, "bert"):
                self._model = class_model.bert
                self._pooler = PoolerAdapter(self.model.pooler)  # ty:ignore[invalid-argument-type]
            elif hasattr(class_model, "roberta"):
                self._model = class_model.roberta
                self._pooler = PoolerAdapter(_cls)  # ty:ignore[invalid-argument-type]
            else:
                raise ValueError(
                    f"Unsupported model {self.model_config.model}: Expected "
                    "Bert or Roberta for sequence classification"
                )
            self.classifier = class_model.classifier

        # Disable pooler because in transformers it's
        # always run even tough we don't use the outputs
        # directly.
        self._model.pooler = None

        model_class_name = type(self.model).__name__
        self.is_roberta = "roberta" in model_class_name.lower()

        self.model.eval()
        torch.set_grad_enabled(False)
        if envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND in BACKEND_LIST:
            # Lazy import to avoid load torch_sendnn runtime before it is really
            # necessary. This solve issues of running forked tests that share
            # some resources from parent to children which can have problems
            # of caching even though the test run in isolated subprocesses.
            SpyrePlatform.maybe_ensure_sendnn_configured(self.model_config)

            with utils_spyre.stagger_region(
                envs_spyre.SENDNN_INFERENCE_MAX_LOAD_PROCESSES,
                self.parallel_config.world_size,
                self.rank,
            ):
                # Not clear how to make the type checking happy with the torch.compile return
                self._model = torch.compile(  # ty: ignore[invalid-assignment]
                    self.model,
                    mode="default",
                    dynamic=False,
                    backend=envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND,
                )

        if task == "classify":
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model)
            output = tokenizer(text="foo", text_pair="bar")
            self.use_token_type_ids = "token_type_ids" in output
            if self.use_token_type_ids:
                self.sep_token_id = tokenizer.sep_token_id

        pooler_config = self.model_config.pooler_config
        assert pooler_config is not None, "Pooler config is require for pooling models"

        if task == "embed":
            with set_current_vllm_config(self.vllm_config):
                self.pooler = pooler_for_embed(pooler_config=pooler_config)
        elif task == "classify":
            with set_current_vllm_config(self.vllm_config):
                self.pooler = pooler_for_classify(
                    pooler_config=pooler_config,
                    pooling=self._pooler,
                    classifier=self.classifier,
                    act_fn=get_act_fn(self.model_config.hf_config),
                )

    @property
    def vocab_size(self) -> int:
        # self.model here is probably a transformers model class
        return self.model.config.vocab_size  # ty: ignore[invalid-return-type]

    def _prepare_pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """left side padding implemented as
        in fms.utils.generation.pad_input_id"""
        max_len = max([min_pad_length] + [seq.size(0) for seq in input_ids_list])
        padded_input_ids_list = []
        mask_list = []
        position_ids_list = []
        for input_ids_i in input_ids_list:
            seq_len = input_ids_i.size(0)
            if max_len > seq_len:
                logger.info(
                    "Left padding request of length %d tokens to %d tokens.", seq_len, max_len
                )
            pads = (
                torch.ones(max_len - seq_len, dtype=torch.long, device=input_ids_i.device)
                * self.pad_token_id
            )
            non_pads = torch.ones(seq_len, dtype=torch.long, device=input_ids_i.device)

            pos_ids_seq = torch.arange(0, seq_len, dtype=torch.long, device=input_ids_i.device)

            # Setting this to 0, however if 0 is the eos, we will end up
            # truncating the output if using truncate_after_eos once this
            # workflow works for nested tensor, this can probably be removed
            padded_input_ids_list.append(torch.cat((pads, input_ids_i)))
            mask_list.append(torch.cat((torch.zeros_like(pads), non_pads)))
            position_ids_list.append(torch.cat((torch.zeros_like(pads), pos_ids_seq)))

        return padded_input_ids_list, mask_list, position_ids_list

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_input_ids_list, mask_list, position_ids_list = self._prepare_pad_input_ids(
            input_ids_list, min_pad_length
        )

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list)
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

    def update_states(self, scheduler_output: SchedulerOutput):
        assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0

        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)
                self.requests.pop(req_id, None)

    def _uncompress_token_types(self) -> list[list[int]]:
        pooling_metadata = self.input_batch.make_pooling_metadata()
        pooling_params = pooling_metadata.pooling_params

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids")) is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return []

        seq_lens = pooling_metadata.prompt_lens
        token_type_ids = []

        for i, seq_len in enumerate(seq_lens):
            pos = token_type_id_requests.get(i, seq_len)
            # TODO: maybe mbayser can see if the ty error here is important
            ids = (torch.arange(seq_lens[i]) >= pos).int()  # ty: ignore
            token_type_ids.append(ids)

        return token_type_ids

    def _token_types(self, input_ids):
        if token_type_ids_lst := self._uncompress_token_types():
            token_type_ids = torch.zeros_like(input_ids)
            for i, token_types in enumerate(token_type_ids_lst):
                token_type_ids[i, -len(token_types) :] = token_types
            return token_type_ids
        else:
            locs = torch.where(input_ids == self.sep_token_id, 1, 0)
            return locs.cumsum(dim=1) - locs

    def _get_padded_batch_size(self, new_requests: list[NewRequestData]):
        # find warmup shape to be used for padding and batching
        applicable_spyre_warmup_shapes = [
            shape for shape in self.spyre_warmup_shapes if len(new_requests) <= shape["batch_size"]
        ]
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids
            updated_spyre_warmup_shapes = [
                shape
                for shape in applicable_spyre_warmup_shapes
                if len(prompt_tokens) <= shape["prompt_length"]  # ty: ignore[invalid-argument-type]
            ]
            applicable_spyre_warmup_shapes = updated_spyre_warmup_shapes

        assert applicable_spyre_warmup_shapes, (
            "No shapes available to run prefill batch. (This should not happen)"
        )

        # If multiple warmup shapes apply, the first one is selected.
        # For improving performance, the warmup shapes in scheduler_config
        # are ordered by "processing speed".
        min_pad_length_batch = applicable_spyre_warmup_shapes[0]["prompt_length"]
        padded_batch_size = applicable_spyre_warmup_shapes[0]["batch_size"]
        return padded_batch_size, min_pad_length_batch

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> PoolingForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []
        padded_batch_size, min_pad_length_batch = self._get_padded_batch_size(new_requests)

        # Internal state is reset here.
        # We don't support continuous batching, so we know all previous requests
        # have finished decoding.
        self.input_batch.clear_requests()
        self.requests = {}

        # Build batch and prepare input_token1
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids
            assert prompt_tokens is not None, "prompt tokens are required for this model runner"

            input_token_list.append(
                torch.tensor(prompt_tokens, dtype=torch.long, device=torch.device("cpu"))
            )

            # Add new requests to the cached states.
            req_id = request_data.req_id
            pooling_params = request_data.pooling_params
            assert pooling_params is not None

            req_state = PoolingRequestState(
                req_id=req_id,
                prompt_token_ids=prompt_tokens,
                pooling_params=pooling_params,
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        self.input_batch.padded_batch_size = padded_batch_size

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch, dtype=torch.long, device=torch.device("cpu"))
            )

        # get position ids and attention mask
        input_tokens, position_ids, mask = self.pad_input_ids(
            input_token_list, min_pad_length=min_pad_length_batch
        )

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = self._token_types(input_tokens)

        if self.is_roberta:
            position_ids += self.pad_token_id + 1
            position_ids *= mask

        model_input = PoolingForwardInputs(
            input_tokens=input_tokens,
            input_embeds=None,
            input_positions=position_ids,
            is_prompt=True,
            input_masks=mask,
            token_type_ids=token_type_ids,
        )

        self._mark_input_tensors(model_input)

        return model_input

    def prepare_model_input(self, scheduler_output: SchedulerOutput) -> PoolingForwardInputs:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        assert is_prompt
        # Assert no running requests
        assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
        return self._prepare_prompt(scheduler_output.scheduled_new_reqs)

    def _mark_input_tensors(self, model_input: PoolingForwardInputs) -> None:
        """Yoinked from
        https://github.com/foundation-model-stack/aiu-fms-testing-utils/pull/13
        """
        if not self.warmup_mode:
            # Only mark tensors when we're warming up and compiling the graphs
            return

        torch._dynamo.mark_static(model_input.input_tokens, 0)
        torch._dynamo.mark_static(model_input.input_tokens, 1)
        torch._dynamo.mark_static(model_input.input_masks, 0)
        torch._dynamo.mark_static(model_input.input_masks, 1)
        torch._dynamo.mark_static(model_input.input_masks, 2)
        torch._dynamo.mark_static(model_input.input_positions, 0)
        torch._dynamo.mark_static(model_input.input_positions, 1)
        if self.use_token_type_ids:
            torch._dynamo.mark_static(model_input.token_type_ids, 0)
            torch._dynamo.mark_static(model_input.token_type_ids, 1)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return tuple(self.pooler.get_supported_tasks())

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        t0 = time.time()

        self.update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        model_input = self.prepare_model_input(scheduler_output)

        attn_metadata = self.build_attn_metadata(model_input)

        model_kwargs = {}
        if self.use_token_type_ids:
            model_kwargs["token_type_ids"] = model_input.token_type_ids

        # Execute the model
        with set_forward_context(attn_metadata, self.vllm_config):
            outputs = self.model(
                input_ids=model_input.input_tokens,
                position_ids=model_input.input_positions,
                attention_mask=model_input.input_masks,
                **model_kwargs,
            )

            hidden_states = outputs["last_hidden_state"]

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return EMPTY_MODEL_RUNNER_OUTPUT

        t1 = time.time() - t0
        logger.debug("t_batch: %.2fms", (t1 * 1000))

        pooling_metadata = self.input_batch.make_pooling_metadata()

        ## No partial prefill, hence we can use the prompt lens here
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np=pooling_metadata.prompt_lens.numpy(),
            seq_lens_cpu=pooling_metadata.prompt_lens,
            device=self.device,
        )

        # prepare unpadded output for the pooler
        hidden_state_list: list[torch.Tensor] = []
        for hidden_state, prompt_len in zip(hidden_states, pooling_metadata.prompt_lens):
            # we're left padding
            hidden_state_list.append(hidden_state[-prompt_len:])

        raw_pooler_output = self.pooler(
            hidden_states=torch.cat(hidden_state_list), pooling_metadata=pooling_metadata
        )

        pooler_output: list[torch.Tensor | None] = []

        for raw_output in raw_pooler_output:
            pooler_output.append(raw_output.data.to("cpu"))

        model_output = ModelRunnerOutput(
            req_ids=self.input_batch.requests_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )
        return model_output


@dataclass
class ChunkedPrefillPlan:
    chunk_count: int
    padding_blocks: int
    usable_cache_blocks: int
    total_cache_blocks: int


class ChunkedPrefillModelRunner(
    BaseSpyreModelRunner[SamplingInputBatch, SamplingRequestState, SamplingForwardInputs]
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        self.block_size = SpyrePlatform.get_block_size()
        self.tkv: int = 0

        self._enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching

        # TODO: Remove this once we can prefill and decode in the same step
        self.prefill_batch = SamplingInputBatch(
            # TODO: review this, currently we only support prefill for
            # `batch_size=1`
            # TODO: when considering multimodal inputs for larger batches, we
            # should also ensure that prefill correctly handles the case in
            # which we mix mm + text-only requests in the same prefill batch.
            max_num_reqs=1,
            max_model_len=vllm_config.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=vllm_config.model_config.get_vocab_size(),
        )

        self.chunk_size = self.scheduler_config.max_num_batched_tokens
        self.chunk_blocks_count = self.chunk_size // self.block_size

        self.prefix_cache_stats = None

        # Initialize performance metric logger for tracking embedding times
        self.perf_logger = create_perf_metric_logger(rank=rank)

    def load_model(self) -> None:
        self._model = SpyreCausalLM(
            vllm_config=self.vllm_config,
            rank=self.rank,
        )

    @property
    def vocab_size(self) -> int:
        model_cfg = self.model.fms_model.config
        if self.model.is_multimodal:
            return self.model.mm_model_utils.resolve_multimodal_vocab_size()
        return model_cfg.src_vocab_size

    @property
    def enable_prefix_caching(self):
        return self._enable_prefix_caching and not self.warmup_mode

    @staticmethod
    def prompt_len(request: NewRequestData | Request) -> int:
        assert request.prompt_token_ids is not None, "prompt token ids are required"
        return len(request.prompt_token_ids)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        # vLLM models have methods available like `is_text_generation_model`
        # We don't use vLLM modeling code though :(
        # Default: assume text generation supported.
        # TODO: Actually detect what the model supports
        tasks.append("generate")

        return tuple(tasks)

    def pre_warmup(self) -> None:
        # Set the number of kv cache blocks to the minimal value of 2 which is
        # required for warmup. After the warmup, the number of blocks will be
        # set to the value returned by the Spyre compiler (see complete_warmup)
        # Note: Until this feature is supported by the compiler we have to set:
        # n_blocks_warmup = n_blocks_avail

        n_blocks_warmup = SpyrePlatform.get_total_spyre_blocks(self.vllm_config)
        # TODO: fixup the typing here. Things are getting tripped up by having all of our "model"
        # classes inherit from `nn.Module` when maybe they don't need to
        self.model.set_past_key_value_states(num_blocks=n_blocks_warmup)

        # Future code:

        # self.model.model.set_past_key_value_states(num_blocks=2)

        # mark the num_blocks dimension dynamic for Spyre compiler for warmup
        # only, compiler will return the number of blocks it can accommodate.
        # (This is not yet supported by the compiler)
        # for layer in self.model.model.past_key_value_states:
        #     for tensor in layer:
        #         torch._dynamo.mark_dynamic(tensor, 0)

    def complete_warmup(self) -> None:
        super().complete_warmup()
        # get the number or pages from the actual Spyre card after the warmup
        # and set it accordingly in the model runner and for the kv cache size
        n_blocks_avail = SpyrePlatform.get_total_spyre_blocks(self.vllm_config)
        # TODO: fixup the typing here. Things are getting tripped up by having all of our "model"
        # classes inherit from `nn.Module` when maybe they don't need to
        self.model.set_past_key_value_states(num_blocks=n_blocks_avail)

    def _get_blocks(self, request_id: str) -> list[int]:
        return self.requests[request_id].block_ids

    def build_attn_metadata(self, model_input: SamplingForwardInputs) -> SpyreAttentionMetadata:
        # TODO: probably we can remove some fields of the model input and
        # update only the SpyreAttentionMetadata

        return SpyreAttentionMetadata(
            slot_mapping=model_input.slot_mapping,
            current_tkv_mask=model_input.current_tkv_mask,
            left_padded_prompt_mask=model_input.left_padded_prompt_mask,
            block_table=model_input.block_table,
            is_prefill=model_input.is_prompt,
        )

    def build_input_batch(self) -> SamplingInputBatch:
        # Define logits processors.

        custom_logitsprocs = self.vllm_config.model_config.logits_processors

        batch_size = self.scheduler_config.max_num_seqs
        logits_processors = build_logitsprocs_for_cb(
            vllm_config=self.vllm_config,
            device=self.device,
            is_pin_memory=self.pin_memory,
            is_pooling_model=False,
            custom_logitsprocs=custom_logitsprocs,
            batch_size=batch_size,
        )

        return SamplingInputBatch(
            max_num_reqs=batch_size,
            max_model_len=self.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            logitsprocs=logits_processors,
        )

    def get_sampling_metadata(self, is_prefill: bool) -> SamplingMetadata:
        if is_prefill:
            sampling_data = self.prefill_batch.sampling_metadata
            sampling_data.logitsprocs = self.input_batch.logitsprocs
            return sampling_data
        else:
            return self.input_batch.sampling_metadata

    def get_req_id_to_index(self, is_prefill: bool) -> dict[str, int]:
        req_id_to_index = (
            self.prefill_batch.get_unpadded_output_indices()
            if is_prefill
            else self.input_batch.get_unpadded_output_indices()
        )

        return req_id_to_index

    def _prepare_prompt(self, new_request_data: list[NewRequestData]) -> SamplingForwardInputs:
        raise NotImplementedError

    def _prepare_chunked_prefill(self, req_id: str) -> SamplingForwardInputs:
        """
        Cases / Scenarios for the chunked prefill with right padding.


        X    - Padding
        T    - Token
        O    - Right padding / unused slot
        |    - Blocks separator
        ||   - Chunks separator
        ...  - Trimmed sequence

        For the following "drawing" consider blocks of 4 tokens
        and chunks of 8 tokens

        # Case I

        Prompt fits in the chunk with left padding

        3 tokens
        1 block
        1 chunk
        4 left padding

        X X X X | T T T O ||

        Variation: Prompt fits in the chunk but no left padding needed

        T tokens
        2 block
        1 chunk
        0 left padding

        T T T T | T T T O ||

        ---
        # Case II

        Prompt is greater than chunk, and it contains left padding

        10 tokens
        4 blocks
        2 chunks
        4 left padding

        X X X X | T T T T || T T T T | T T O O ||

        # Case III

        No left padding and more than one chunk

        13 tokens
        4 blocks
        2 chunks
        0 left padding

        T T T T | T T T T || T T T T | T O O O ||

        NOTE: The goal of this "illustration" is to depict strategies to write
        code to create the chunks, not necessarily enumerate the possible
        scenarios. Of course there are interpretations where these cases
        overlap.

        """
        request = self.requests[req_id]

        chunk_size = self.chunk_size
        left_padding = request.padding_blocks * self.block_size
        left_padded_prompt_mask = torch.tensor(
            [left_padding], dtype=torch.int64, device=self.device
        )

        mm_features = getattr(request, "mm_features", None)

        # we need to figure out what chunk we're processing. vLLM will only
        # schedule from the first chunk that contains a miss or must
        # be recomputed onwards.
        chunk_i = math.floor(
            (request.padding_blocks * self.block_size + request.num_computed_tokens)
            / self.chunk_size
        )
        assert chunk_i < request.chunk_count

        # create block table tensor
        block_end = (chunk_i + 1) * self.chunk_blocks_count
        block_ids = [0] * request.padding_blocks + self._get_blocks(req_id)
        block_table = torch.tensor(block_ids[:block_end], dtype=torch.int64).unsqueeze(0)

        # last chunk
        blocks_to_recompute = 0
        if request.total_hit_blocks > 0:
            if request.usable_blocks == 0:
                chunks_from_cache = 0
            else:
                chunks_from_cache = exact_div(
                    request.padding_blocks + request.usable_blocks, self.chunk_blocks_count
                )

            # When the current chunk has passed the number
            # of chunk loaded entirely from cache, the difference
            # between the blocks from cache and the allocated
            # blocks will the the number of blocks to recompute.
            if chunk_i == chunks_from_cache:
                blocks_to_recompute = request.total_hit_blocks - request.usable_blocks
                # For the first block we must also account for left-padding blocks:
                # blocks_to_recompute = total_hit_blocks - usable_blocks + padding_blocks
                if chunk_i == 0:
                    blocks_to_recompute += request.padding_blocks

        slot_mapping = []
        for i in range(self.chunk_blocks_count):
            block = int(block_table[0][-self.chunk_blocks_count + i].item())
            # if we're recomputing a cached block, set the slot
            # mapping to the padding block (0)
            block *= int(i >= blocks_to_recompute)
            slot_mapping += list(
                range(block * self.block_size, block * self.block_size + self.block_size)
            )
        slot_mapping = torch.tensor(slot_mapping, device=self.device, dtype=torch.int64).unsqueeze(
            0
        )

        prompt_token_ids = request.prompt_token_ids
        prompt_len = len(prompt_token_ids)
        if chunk_i == 0:
            chunk_start = 0
            chunk_end = min(chunk_size - left_padding, prompt_len)
            chunk_left_offset = left_padding
        else:
            chunk_start = chunk_i * chunk_size - left_padding
            chunk_end = min(chunk_start + chunk_size, prompt_len)
            chunk_left_offset = 0

        input_tokens = torch.zeros(chunk_size, dtype=torch.int64, device=self.device)
        input_tokens_np = input_tokens.numpy()
        input_positions = torch.zeros(chunk_size, dtype=torch.int64, device=self.device)
        input_positions_np = input_positions.numpy()

        # Create tensors based on slice
        input_tokens_np[chunk_left_offset : chunk_left_offset + chunk_end - chunk_start] = (
            prompt_token_ids[chunk_start:chunk_end]
        )
        input_positions_np[chunk_left_offset : chunk_left_offset + chunk_end - chunk_start] = range(
            chunk_start, chunk_end
        )

        logger.debug(
            "Chunked prefill of request '%s' %d:%d of %d tokens",
            req_id,
            chunk_start,
            chunk_end,
            prompt_len,
        )

        input_tokens = input_tokens.unsqueeze(0).clone()
        input_positions = input_positions.unsqueeze(0).clone()

        # NOTE(wallas): Looks like we need to use multiple of blocks for prefill
        # so, later we use model.n_pads_right to get right logits.
        # In my naive mind this would be the `request_tkv` below,
        # but it gives me incorrect results
        #
        prefill_tkv = (chunk_i + 1) * chunk_size
        current_tkv_mask = torch.tensor([prefill_tkv], dtype=torch.int64, device=self.device)

        request_tkv = min(prefill_tkv, left_padding + prompt_len)

        # Trick padding:
        # In `model_executor/model_loader/spyre.py`: We have this line to
        # get the logits of a prefill:
        #
        #   logits = logits[self.indices, -self.n_pads_right - 1, :]
        #
        # So we have to match this padding with the tkv of this chunk
        # being prefilled.
        # `((request_tkv -1) % self.block_size) + 1`: gives us the tkv offset
        # removing all the left blocks. Reminder: this tkv must be in the range
        # [1, block_size]
        #
        # `self.block_size - `: will just flip the index to get it as
        # negative index
        self.model.n_pads_right = self.block_size - (((request_tkv - 1) % self.block_size) + 1)
        self.model.indices = torch.ones(1, dtype=torch.bool, device="cpu")

        # For multimodal requests, compute embeddings once for the full sequence
        # and cache them, then slice per chunk. This ensures image features are
        # correctly aligned across all chunks.
        if mm_features and request.cached_mm_embeddings is None:
            # First chunk: compute full multimodal embeddings
            full_input_tokens = torch.tensor(
                prompt_token_ids, dtype=torch.int64, device=self.device
            ).unsqueeze(0)

            t0 = time.time()
            full_embeds = self.model.get_maybe_mm_embeddings(
                full_input_tokens,
                mm_features=mm_features,
                is_decode=False,
            )

            t_elapsed = time.time() - t0

            logger.info("maybe_mm_embedding processing time: %.2fms", (t_elapsed * 1000))
            self.perf_logger.log(
                "get_mm_embeddings_time_ms",
                t_elapsed * 1000,
                phase="prefill",
                has_mm_features=True,
                req_id=req_id,
            )

            # Cache the full embeddings for subsequent chunks
            request.cached_mm_embeddings = full_embeds
            logger.debug("Computed and cached full multimodal embeddings for request '%s'", req_id)

        # Slice the cached embeddings for this chunk
        if request.cached_mm_embeddings is not None:
            # Extract the slice corresponding to this chunk
            # Add left padding to align with the chunked token positions
            full_embeds = request.cached_mm_embeddings

            # Create output tensor with chunk size
            input_embeds = torch.zeros(
                (1, chunk_size, full_embeds.shape[-1]),
                dtype=full_embeds.dtype,
                device=self.device,
            )

            # Copy the relevant slice from full embeddings
            # This mirrors the logic for input_tokens slicing above
            input_embeds[0, chunk_left_offset : chunk_left_offset + chunk_end - chunk_start] = (
                full_embeds[0, chunk_start:chunk_end]
            )

            logger.debug(
                "Sliced embeddings for chunk %d of request '%s': [%d:%d] -> [%d:%d]",
                chunk_i,
                req_id,
                chunk_start,
                chunk_end,
                chunk_left_offset,
                chunk_left_offset + chunk_end - chunk_start,
            )
        else:
            # Non-multimodal or decode: use standard token embedding
            t0 = time.time()
            input_embeds = self.model.get_maybe_mm_embeddings(
                input_tokens,
                mm_features=None,
                is_decode=False,
            )
            t_elapsed = time.time() - t0
            self.perf_logger.log(
                "get_mm_embeddings_time_ms",
                t_elapsed * 1000,
                phase="prefill",
                has_mm_features=False,
                req_id=req_id,
            )

        model_inputs = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_embeds=input_embeds,
            input_positions=input_positions,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=True,
        )

        self._mark_input_tensors(model_inputs)

        return model_inputs

    def _prepare_decode(
        self,
        cached_request_data: CachedRequestData,
    ) -> SamplingForwardInputs:
        assert len(cached_request_data.req_ids) > 0

        input_tokens = []
        input_positions = []
        block_table = []
        slot_mapping = []
        left_padded_prompt_mask = []
        tkv_mask = []

        assert len(self.input_batch.req_id_to_index) == len(cached_request_data.req_ids)
        req_ids = self.input_batch.sorted_requests_ids

        # maximal number of blocks used by any seq in the batch
        max_n_blocks = 0

        for req_id in req_ids:
            max_n_blocks = max(max_n_blocks, len(self._get_blocks(req_id)))

        # We'll calculate tkv on the fly, it is the max num computed tokens
        # of a request since there is no tokens left padding, only for blocks
        tkv = 0
        for req_id in req_ids:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?

            req_state = self.requests[req_id]

            # filling block table with padding blocks to make it rectangular
            blocks = self._get_blocks(req_id)
            left_pad_blocks_count = max_n_blocks - len(blocks)
            block_ids = [0] * left_pad_blocks_count + blocks
            block_table.append(block_ids)
            # Update the internal request state with the number of padding blocks used
            req_state.padding_blocks = left_pad_blocks_count

            # slot_mapping for all blocks of sequence
            start_slot = block_table[-1][-1] * self.block_size
            offset = req_state.num_computed_tokens % self.block_size
            slot = [start_slot + offset]
            slot_mapping.append(slot)

            # input token and position of the token generated in the last step
            generation_token = req_state.output_token_ids[-1]
            input_tokens.append([generation_token])
            input_positions.append([req_state.num_computed_tokens])

            # Calculate left padding on the fly
            left_padding = left_pad_blocks_count * self.block_size
            left_padded_prompt_mask.append(left_padding)

            req_tkv = left_padding + req_state.num_computed_tokens + 1
            tkv_mask.append(req_tkv)
            tkv = max(tkv, req_tkv)

        # update tkv
        self.tkv = tkv

        # construct tensors from lists
        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(input_positions, dtype=torch.long, device=self.device)
        current_tkv_mask = torch.tensor(tkv_mask, dtype=torch.int64)
        left_padded_prompt_mask = torch.tensor(
            left_padded_prompt_mask, dtype=torch.long, device=self.device
        )
        block_table = torch.tensor(block_table, dtype=torch.int64)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        self.model.indices = torch.ones(
            len(cached_request_data.req_ids), dtype=torch.bool, device="cpu"
        )

        # None unless this model is multimodal; no mm_features since
        # all multimodal features are merged in prefill.
        t0 = time.time()
        input_embeds = self.model.get_maybe_mm_embeddings(
            input_tokens,
            mm_features=None,
            is_decode=True,
        )
        t_elapsed = time.time() - t0
        # Log timing for each request in the decode batch
        for req_id in cached_request_data.req_ids:
            self.perf_logger.log(
                "get_mm_embeddings_time_ms",
                t_elapsed * 1000,
                phase="decode",
                has_mm_features=False,
                req_id=req_id,
            )

        model_inputs = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_embeds=input_embeds,
            input_positions=position_ids,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=False,
        )

        self._mark_input_tensors(model_inputs)

        return model_inputs

    def _plan_chunking(
        self, prompt_token_ids: list[int], num_computed_tokens: int
    ) -> ChunkedPrefillPlan:
        prompt_len = len(prompt_token_ids)

        chunk_size = self.chunk_size
        padded_prompt_len = math.ceil(prompt_len / self.block_size) * self.block_size
        chunk_count = math.ceil(prompt_len / chunk_size)

        left_padding = chunk_count * chunk_size - padded_prompt_len
        left_blocks = exact_div(left_padding, self.block_size)

        if self.enable_prefix_caching:
            n_hit = exact_div(num_computed_tokens, self.block_size)

            logger.debug("Prefix caching found: %d cached blocks", n_hit)

            full_chunks_with_cached_blocks = (left_blocks + n_hit) // self.chunk_blocks_count

            # the last chunk of the prompt must always be recomputed
            if full_chunks_with_cached_blocks == chunk_count:
                full_chunks_with_cached_blocks -= 1

            usable_blocks = max(
                0, (full_chunks_with_cached_blocks * self.chunk_blocks_count) - left_blocks
            )

            # blocks to compute from scratch or recompute in the last chunk
            blocks_to_compute = padded_prompt_len // self.block_size - usable_blocks
            logger.debug(
                "Prefix caching found: %d reusable blocks in cache. %d blocks will be (re)computed",
                usable_blocks,
                blocks_to_compute,
            )

        else:
            usable_blocks = 0
            n_hit = 0

        return ChunkedPrefillPlan(
            chunk_count=chunk_count,
            padding_blocks=left_blocks,
            usable_cache_blocks=usable_blocks,
            total_cache_blocks=n_hit,
        )

    def add_new_request(self, request: NewRequestData):
        req_id = request.req_id
        prompt_token_ids = request.prompt_token_ids
        sampling_params = request.sampling_params

        assert sampling_params is not None, "sampling_params are required for this model runner"
        assert prompt_token_ids is not None, "prompt token ids are required for this model runner"

        is_new_batch = self.input_batch.num_reqs == 0
        prompt_len = len(prompt_token_ids)
        mm_features = getattr(request, "mm_features", None)

        self.prefill_batch.clear_requests()

        # set the new tkv to the prompt length if starting a new decode batch
        if is_new_batch:
            self.tkv = prompt_len

        block_ids_per_kv_cache_group = request.block_ids
        assert len(block_ids_per_kv_cache_group) == 1

        chunk_plan = self._plan_chunking(prompt_token_ids, request.num_computed_tokens)

        # Add new request to the cached states.
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            seed = sampling_params.seed if sampling_params.seed is not None else 0
            generator.manual_seed(seed)
        else:
            generator = None

        req_state = SamplingRequestState(
            generator=generator,
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            mm_features=mm_features,
            num_computed_tokens=request.num_computed_tokens,
            chunk_count=chunk_plan.chunk_count,
            padding_blocks=chunk_plan.padding_blocks,
            usable_blocks=chunk_plan.usable_cache_blocks,
            total_hit_blocks=chunk_plan.total_cache_blocks,
            block_ids=request.block_ids[0],  # we only support on kv cache group for now
        )

        self.requests[req_id] = req_state

        # Add only to prefill batch, it will be added later to the input batch
        # once if is fully prefilled
        self.prefill_batch.add_request(req_state)

    def _maybe_prepare_last_prefill(self, req_id: str, scheduler_output: SchedulerOutput) -> None:
        """In the last prefill we have to setup the batch to sample the
        first token.
        """
        # Check if it is last prefill
        request = self.requests[req_id]
        num_computed_tokens = request.num_computed_tokens
        prompt_len = len(request.prompt_token_ids)
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

        if num_computed_tokens + num_scheduled_tokens < prompt_len:
            return

        # Last prefill: clear cached multimodal embeddings to free memory
        if request.cached_mm_embeddings is not None:
            logger.debug(
                "Clearing cached multimodal embeddings for request '%s' after last prefill",
                req_id,
            )
            request.cached_mm_embeddings = None

        # Last prefill: we might need to update the tkv
        req_n_blocks = math.ceil(prompt_len / self.block_size)
        cur_n_blocks = math.ceil(self.tkv / self.block_size)
        new_n_blocks = max(req_n_blocks, cur_n_blocks)
        assert new_n_blocks > 0
        base_n_tokens = (new_n_blocks - 1) * self.block_size
        req_tkv_new_block = base_n_tokens + (prompt_len - 1) % self.block_size + 1
        cur_tkv_new_block = base_n_tokens + (self.tkv - 1) % self.block_size + 1
        self.tkv = max(req_tkv_new_block, cur_tkv_new_block)

        # Last prefill we need to setup the logitsprocessors to sampling
        prefill_index = self.input_batch.add_request(request)
        for logitsproc in self.input_batch.logitsprocs_wrappers:
            logitsproc.set_prefill_index(prefill_index)

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_metadata()
        self.prefill_batch.refresh_metadata()

    def prepare_model_input(self, scheduler_output: SchedulerOutput) -> SamplingForwardInputs:
        is_prefill = False
        req_id: str = ""
        if len(scheduler_output.scheduled_new_reqs) == 1:
            # First prefill let's update cache
            assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
            req_id = scheduler_output.scheduled_new_reqs[0].req_id
            is_prefill = True

        # NOTE: We assume that there's only one prefill at each step
        # and if it is a single request cached here and the number of
        # computed tokens is less than the length of the prompt then
        # it is still prefilling.
        if scheduler_output.scheduled_cached_reqs.num_reqs == 1:
            # Whether it's a prefill or not, should not have any request here
            assert len(scheduler_output.scheduled_new_reqs) == 0
            req_id = scheduler_output.scheduled_cached_reqs.req_ids[0]
            is_prefill = (
                len(self.requests[req_id].prompt_token_ids)
                > scheduler_output.scheduled_cached_reqs.num_computed_tokens[0]
            )

        # Prepare input tensors.
        if is_prefill:
            # All prefills are chunked
            # Get request id from new request or cached request
            model_inputs = self._prepare_chunked_prefill(req_id)
            self._maybe_prepare_last_prefill(req_id, scheduler_output)

            return model_inputs
        else:
            return self._prepare_decode(scheduler_output.scheduled_cached_reqs)

    def get_empty_output(self):
        return SpyreModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            num_nans_in_logits=None,
            tkv=0,
            left_padding={},
        )

    def check_incomplete_prefill(self, scheduler_output: SchedulerOutput):
        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs

        if cached_reqs.num_reqs != 1 and len(new_reqs) != 1:
            # Not a prefill
            return False

        # possible prefill
        req_id = new_reqs[0].req_id if len(new_reqs) == 1 else cached_reqs.req_ids[0]

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if len(new_reqs) == 1:
            num_computed_tokens = new_reqs[0].num_computed_tokens
            return (num_computed_tokens + num_scheduled_tokens) < self.prompt_len(new_reqs[0])
        else:
            req_state = self.requests[req_id]
            num_computed_tokens = cached_reqs.num_computed_tokens[0]
            return (num_computed_tokens + num_scheduled_tokens) < len(req_state.prompt_token_ids)

    def update_states(self, scheduler_output: SchedulerOutput):
        # clear the prefix cache stats so that we only record them on the first
        # chunk of any prefill
        self.prefix_cache_stats = None
        self._update_batch(scheduler_output)

    def _update_batch(self, scheduler_output: SchedulerOutput):
        """Updates the states for the in progress batch
        - Bumps the count of computed tokens for each request
        - Updates the KV cache metadata for each request
        - Safely removes finished requests from the batch
        - Refreshes metadata for logits processors
        """
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state: SamplingRequestState = self.requests[req_id]

            new_block_ids_per_kv_cache_group = req_data.new_block_ids[i]
            if new_block_ids_per_kv_cache_group:
                assert len(new_block_ids_per_kv_cache_group) == 1
                req_state.block_ids.extend(new_block_ids_per_kv_cache_group[0])

            # Update the number of computed tokens for this request
            num_computed_tokens = req_data.num_computed_tokens[i]
            req_state.num_computed_tokens = num_computed_tokens

        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)
                # TODO: Processing multiple removals at once can break alignment
                # of logitprocs. Refactor so that we can batch removals to the
                # `input_batch`
                self.input_batch.refresh_metadata()
        else:
            # Due to logits processor we need to refresh metadata at each step
            self.input_batch.refresh_metadata()

    def maybe_setup_new_prefill(self, scheduler_output: SchedulerOutput):
        """If this schedule is for the first chunk of a new prefill, set up the internal state
        appropriately. This updates self.prefill_batch and self.requests, calculating the prefix
        cache hit.
        """
        if len(scheduler_output.scheduled_new_reqs) > 0:
            # A request is only here when the very first chunk is scheduled
            assert len(scheduler_output.scheduled_new_reqs) == 1, (
                "Can only schedule one chunked prefill at a time"
            )
            assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0, (
                "Cannot schedule a new prefill and running requests in the same execution"
            )
            self.add_new_request(scheduler_output.scheduled_new_reqs[0])

    def is_cached_chunk(self, scheduler_output: SchedulerOutput):
        """Returns true iff this schedule is for one chunk of a prefill, and that chunk is fully
        cached in the prefix cache."""
        if len(scheduler_output.scheduled_new_reqs) == 1:
            req_id = scheduler_output.scheduled_new_reqs[0].req_id
        elif len(scheduler_output.scheduled_cached_reqs.req_ids) == 1:
            req_id = scheduler_output.scheduled_cached_reqs.req_ids[0]
        else:
            # Not a prefill
            return False

        request = self.requests[req_id]
        num_computed_tokens = request.num_computed_tokens
        num_computed_blocks = exact_div(num_computed_tokens, self.block_size)

        if request.usable_blocks > num_computed_blocks:
            assert self.enable_prefix_caching, "Prefix caching must be enabled"
            return True
        return False

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
        batch: SamplingInputBatch,
    ) -> None:
        """Apply grammar bitmask in-place to constrain logits for structured
        output requests.
        """
        grammar_output = getattr(scheduler_output, "_spyre_grammar_output", None)
        if grammar_output is None:
            return

        vllm_apply_grammar_bitmask(
            scheduler_output,
            grammar_output,
            batch,  # type: ignore[arg-type]
            logits,
        )

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        t0 = time.time()

        self.update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return self.get_empty_output()

        # Initialize internal request states if this is the first chunk of a very new prefill
        self.maybe_setup_new_prefill(scheduler_output)

        model_input = self.prepare_model_input(scheduler_output)
        is_prefill = model_input.is_prompt

        # Execute the model
        attn_metadata = self.build_attn_metadata(model_input)
        # Embeddings take priority [used by multimodal models only]
        input_ids_or_embeds = (
            model_input.input_embeds
            if model_input.input_embeds is not None
            else model_input.input_tokens
        )

        with set_forward_context(attn_metadata, self.vllm_config):
            assert (
                self.tkv * len(scheduler_output.num_scheduled_tokens)
                <= SpyrePlatform.get_max_batch_tkv_limit()
            ), (
                f"Exceeded max batch tkv limit {SpyrePlatform.get_max_batch_tkv_limit()}!"
                f" tkv: {self.tkv}, batch_size: {len(scheduler_output.num_scheduled_tokens)}"
            )

            logits = self.model(
                input_ids_or_embeds=input_ids_or_embeds,
                positions=model_input.input_positions,
                masks=None,
                is_prompt=model_input.is_prompt,
            )

        # If the prompt is being prefilled we don't have to sample
        # and generate a new token.
        if is_prefill and self.check_incomplete_prefill(scheduler_output):
            # Only return outputs from the driver worker
            if not self.is_driver_worker:
                return self.get_empty_output()

            t1 = time.time() - t0
            logger.debug("t_forward_pass: %.2fms [prefill single chunk][batch size 1]", (t1 * 1000))
            return self.prefill_output()

        # Apply grammar bitmask for structured output requests.
        self.apply_grammar_bitmask(
            scheduler_output,
            logits,
            self.prefill_batch if is_prefill else self.input_batch,
        )

        # Sample the next token.
        output: SamplerOutput | None = self.model.sample(
            logits=logits,
            sampling_metadata=self.get_sampling_metadata(is_prefill),
        )
        assert output is not None, "Expected sampler output"

        t1 = time.time() - t0
        batch_size = model_input.input_tokens.shape[0]
        step_type = "[prefill last chunk]" if is_prefill else "[decode]"
        logger.debug("t_token: %.2fms %s[batch size %d]", (t1 * 1000), step_type, batch_size)

        # Get the right batch, if this is the last chunk to conclude the
        # prefill, we'll generate a token and we should get from the prefill
        # batch because input_batch may have other request that are were
        # not processed at this step.
        batch = self.prefill_batch if is_prefill else self.input_batch

        # Add the sampled token(s) to the request cache
        req_ids = (
            [r.req_id for r in scheduler_output.scheduled_new_reqs]
            if len(scheduler_output.scheduled_new_reqs) > 0
            else batch.sorted_requests_ids
        )
        sampled_ids = output.sampled_token_ids.tolist()

        for i, req_id in enumerate(req_ids):
            req_state = self.requests[req_id]
            req_state.append_output_token_ids(sampled_ids[i])

        # Only return outputs from the driver worker
        if not self.is_driver_worker:
            return self.get_empty_output()

        model_output = self.sampled_output(output, is_prefill)
        return model_output

    def prefill_output(self) -> SpyreModelRunnerOutput:
        req_id_to_index = self.get_req_id_to_index(is_prefill=True)
        left_padding = {
            req_id: self.requests[req_id].padding_blocks * self.block_size
            for req_id in req_id_to_index
        }

        return SpyreModelRunnerOutput(
            req_ids=list(req_id_to_index.keys()),
            req_id_to_index=req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            tkv=self.tkv,
            left_padding=left_padding,
            prefix_cache_hit_len=self.get_prefix_cache_len(),
        )

    def sampled_output(self, output: SamplerOutput, is_prefill: bool) -> SpyreModelRunnerOutput:
        req_id_to_index = self.get_req_id_to_index(is_prefill)
        left_padding = {
            req_id: self.requests[req_id].padding_blocks * self.block_size
            for req_id in req_id_to_index
        }

        return SpyreModelRunnerOutput(
            req_ids=list(req_id_to_index.keys()),
            req_id_to_index=req_id_to_index,
            sampled_token_ids=output.sampled_token_ids.tolist(),
            logprobs=(output.logprobs_tensors.tolists() if output.logprobs_tensors else None),
            prompt_logprobs_dict={},
            pooler_output=[],
            tkv=self.tkv,
            left_padding=left_padding,
        )

    def get_prefix_cache_len(self) -> dict[str, int]:
        """Get the prefix cache hit length for each prefilling request.
        This is in the number of usable cache tokens: Including the left padding
        this will always land at a chunk boundary.
        """
        result = {}
        for req_id in self.prefill_batch.requests_ids:
            request = self.requests[req_id]
            result[req_id] = request.usable_blocks * self.block_size
        return result

    def _mark_input_tensors(self, model_input: SamplingForwardInputs) -> None:
        # Marking dimensions static/dynamic
        if model_input.is_prompt:
            # batch static (batch size 1)
            torch._dynamo.mark_static(model_input.slot_mapping, 0)
            torch._dynamo.mark_static(model_input.input_positions, 0)
            torch._dynamo.mark_static(model_input.block_table, 0)

            # sequence dynamic
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 1)
            torch._dynamo.mark_dynamic(model_input.input_positions, 1)
            torch._dynamo.mark_dynamic(model_input.block_table, 1)

            # In the case that the input tokens are 3D, i.e., they're actually
            # embeddings, The last dimension (embedding dimension) is static.
            # This is mostly for multimodal models.
            if model_input.input_embeds is not None:
                torch._dynamo.mark_static(model_input.input_embeds, 0)
                torch._dynamo.mark_dynamic(model_input.input_embeds, 1)
                torch._dynamo.mark_static(model_input.input_embeds, 2)
            else:
                torch._dynamo.mark_static(model_input.input_tokens, 0)
                torch._dynamo.mark_dynamic(model_input.input_tokens, 1)

        # decode
        else:
            # mask is no longer used here

            # batch dynamic
            torch._dynamo.mark_dynamic(model_input.block_table, 0)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 0)
            torch._dynamo.mark_dynamic(model_input.input_positions, 0)
            torch._dynamo.mark_dynamic(model_input.current_tkv_mask, 0)
            torch._dynamo.mark_dynamic(model_input.left_padded_prompt_mask, 0)

            # sequence
            torch._dynamo.mark_dynamic(model_input.block_table, 1)
            torch._dynamo.mark_static(model_input.slot_mapping, 1)  # always 1
            torch._dynamo.mark_static(model_input.input_positions, 1)  # always 1

            if model_input.input_embeds is not None:
                torch._dynamo.mark_dynamic(model_input.input_embeds, 0)
                torch._dynamo.mark_static(model_input.input_embeds, 1)  # always 1
                torch._dynamo.mark_static(model_input.input_embeds, 2)
            else:
                torch._dynamo.mark_dynamic(model_input.input_tokens, 0)
                torch._dynamo.mark_static(model_input.input_tokens, 1)  # always 1
