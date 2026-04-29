"""Utilities for selecting and loading Spyre models."""

import os
from dataclasses import dataclass
from typing import cast

from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor
import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
from fms.models import get_model
from transformers import PretrainedConfig, Mistral3Config
from vllm.config import ModelConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

import sendnn_inference.envs as envs_spyre
import sendnn_inference.multimodal as spyre_mm
import sendnn_inference.utils as utils_spyre
from sendnn_inference.platform import SpyrePlatform

try:
    import backends.dynamo_tracer  # ty: ignore[unresolved-import] # noqa
except ImportError:
    print("WARNING: Disabled: dynamo_tracer")
    pass

BACKEND_LIST = ["sendnn", "sendnn_compile_only", "inductor"]

logger = init_logger(__name__)


@dataclass
class SpyreAttentionMetadata:
    slot_mapping: torch.Tensor
    current_tkv_mask: torch.Tensor
    left_padded_prompt_mask: torch.Tensor
    block_table: torch.Tensor
    is_prefill: bool


class SpyreCausalLM(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
    ) -> None:
        super().__init__()

        self.sampler = Sampler()

        # boolean tensor of length batch size with indices:
        # True for unfinished sequences and
        # False for finished or padded sequences
        self.indices = torch.ones(0)

        # number of right pads
        self.n_pads_right = 0

        self.on_spyre = SpyrePlatform.is_backend_sendnn_enabled()
        self._mask_dtype = torch.float16 if self.on_spyre else torch.float32

        self.config = self.resolve_hf_config(vllm_config)

        # Actual FMS model
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.dtype = self.get_dtype()

        # Wrappers for utils for multimodal
        self.mm_model_utils: spyre_mm.MMUtilsBase | None = None
        self.is_multimodal = False

        BLOCK_SIZE = SpyrePlatform.get_block_size()
        max_model_len = vllm_config.model_config.max_model_len

        # edge case: prompt fills model length: can produce 1 token with prefill
        max_prompt_length = max_model_len
        # edge case: prompt will be padded to first block:
        # can produce 1 token with prefill plus rest of model length
        max_decode_length = max_model_len - BLOCK_SIZE + 1

        # Load the weights from the cached or downloaded files.
        self.load_weights(
            model_config=self.model_config,
            max_prompt_length=max_prompt_length,
            max_decode_length=max_decode_length,
            distributed_strategy="tp" if self.parallel_config.world_size > 1 else None,
            sendnn_dynamic=self.on_spyre,
            rank=rank,
            world_size=self.parallel_config.world_size,
        )

        # physical KV cache on AIU Spyre: will eventually not live in this class
        self.kv_cache_specs = {}
        self.kv_cache_specs["block_size"] = BLOCK_SIZE
        self.kv_cache_specs["num_kv_heads"] = self.model_config.get_num_kv_heads(
            self.parallel_config
        )

        if self.config.model_type in {"llama", "granite", "granitemoehybrid"}:
            self.kv_cache_specs["num_layers"] = self.config.num_hidden_layers
            self.kv_cache_specs["head_dim"] = getattr(
                self.fms_model.config,
                "head_dim",
                self.config.hidden_size // self.config.num_attention_heads,
            )
        elif self.config.model_type == "gpt_bigcode":
            self.kv_cache_specs["num_layers"] = self.config.n_layer
            self.kv_cache_specs["head_dim"] = self.config.n_embd // self.config.n_head
        elif self.is_multimodal and self.mm_model_utils is not None:
            # Handle multimodal separately for now since we need to unwrap the
            # text configs and technically (outside FMS) the LLM could be
            # generic; the instance of mm_model_utils encapsulates the configs,
            # so no need to pass them again.
            unwrapped_opts = self.mm_model_utils.unwrap_mm_kv_cache_opts()
            self.kv_cache_specs.update(unwrapped_opts)
        else:
            raise NotImplementedError(
                f"[SpyreCausalLM] model type {self.config.model_type} "
                f"not supported in ContinuousBatchingFmsModel"
            )

        if self.model_config.quantization:
            self.attention_name = "spyre_paged_attn_fp8"
            self.is_fp8_model = True
        else:
            self.attention_name = "spyre_paged_attn"
            self.is_fp8_model = False

        self.current_scale: list[tuple] | None = None
        self.past_key_value_states: list[
            tuple[torch.Tensor | ScaledTensor, torch.Tensor | ScaledTensor]
        ] = []

    def load_weights(
        self,
        model_config: ModelConfig,
        max_prompt_length: int,
        max_decode_length: int,
        distributed_strategy: str | None,
        sendnn_dynamic: bool,
        **kwargs,
    ) -> None:
        logger.debug("Loading model weights for model %s", model_config.model)
        logger.debug("Model config has dtype: %s", model_config.dtype)

        # When using quantized models, we might not be using the
        # model_config's dtype, hence we don't log the msg below
        # since it might confuse the user
        if model_config.quantization:
            logger.debug("Quantized model found with quantization : %s", model_config.quantization)
        else:
            if self.dtype is not model_config.dtype:
                logger.info(
                    "Ignoring user-provided dtype=%s (provided either through"
                    " --dtype CLI arg or model_config.dtype) and using"
                    " dtype=%s instead.",
                    model_config.dtype,
                    self.dtype,
                )

        is_local = os.path.isdir(model_config.model)
        model_path = model_config.model
        # Get location of model from HF cache.
        if not is_local:
            model_path = download_weights_from_hf(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=["*.safetensors", "*.bin", "*.pt"],
                revision=model_config.revision,
            )

        # Get any fixes needed that must be patched into the kwargs;
        # currently this is only use for multimodal models / llava next
        model_kwargs = spyre_mm.get_mm_specific_load_overrides(self.config)

        with utils_spyre.stagger_region(
            envs_spyre.SENDNN_INFERENCE_MAX_LOAD_PROCESSES,
            kwargs["world_size"],
            kwargs["rank"],
        ):
            self.fms_model = get_model(
                architecture="hf_pretrained",
                model_path=model_path,
                distributed_strategy=distributed_strategy,
                group=dist.group.WORLD,
                fused_weights=False,
                trust_remote_code=model_config.trust_remote_code,
                **model_kwargs,
            )

        self.fms_model.eval()
        torch.set_grad_enabled(False)

        _target_cache_size = max(int(max_decode_length * 2), int(max_prompt_length * 2.5))
        if (
            hasattr(torch._dynamo.config, "accumulated_cache_size_limit")
            and _target_cache_size > torch._dynamo.config.accumulated_cache_size_limit
        ):
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = _target_cache_size
            logger.info(
                "NOTICE: Adjusting "
                "torch._dynamo.config.accumulated_cache_size_limit "
                "from %s to %s "
                "to accommodate prompt size of %d "
                "and decode tokens of %d",
                _prev,
                torch._dynamo.config.accumulated_cache_size_limit,
                max_prompt_length,
                max_decode_length,
            )

        if _target_cache_size > torch._dynamo.config.cache_size_limit:
            _prev = torch._dynamo.config.cache_size_limit
            torch._dynamo.config.cache_size_limit = _target_cache_size
            logger.info(
                "NOTICE: Adjusting torch._dynamo.config.cache_size_limit "
                "from %s to %s "
                "to accommodate prompt size of %d "
                "and decode tokens of %d",
                _prev,
                torch._dynamo.config.accumulated_cache_size_limit,
                max_prompt_length,
                max_decode_length,
            )

        # Initialize mm utils before the cast (reads mm_parameter_prefixes)
        # and before torch.compile wraps fms_model (reads fms_model.config).
        self.mm_model_utils = spyre_mm.maybe_get_mm_utils(
            model_path=model_path,
            fms_config=self.fms_model.config,
            hf_config=self.config,
        )
        self.is_multimodal = self.mm_model_utils is not None

        if envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND in BACKEND_LIST:
            self._cast_params_for_spyre()
            options = {"sendnn.dynamic": True} if sendnn_dynamic else {}

            # Lazy import to avoid load torch_sendnn runtime before it is really
            # necessary. This solve issues of running forked tests that share
            # some resources from parent to children which can have problems
            # of caching even though the test run in isolated subprocesses.
            SpyrePlatform.maybe_ensure_sendnn_configured(self.model_config)

            self.fms_model = torch.compile(
                self.fms_model,
                backend=envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND,
                options=options,
            )
        else:
            # CPU execution
            # For continuous batching w/ paged attention, we only support either
            # fp32 or fp8, not f16 or bf16.
            if not model_config.quantization:
                assert self.dtype == torch.float32
                self._cast_to_f32()

        logger.debug("Model weights loaded successfully.")

    def _cast_params_for_spyre(self):
        """Cast mm params to SENDNN_INFERENCE_CPU_MM_DTYPE; remaining bf16 params to fp16."""
        cpu_mm_dtype = envs_spyre.SENDNN_INFERENCE_CPU_MM_DTYPE
        mm_prefixes = self.mm_model_utils.mm_parameter_prefixes if self.mm_model_utils else ()

        for name, param in self.fms_model.named_parameters():
            if name.startswith(mm_prefixes):
                if param.dtype != cpu_mm_dtype:
                    logger.debug(
                        "Casting vision encoder param %s to %s for CPU execution.",
                        name,
                        cpu_mm_dtype,
                    )
                    param.data = param.data.to(dtype=cpu_mm_dtype)
            elif param.dtype == torch.bfloat16:
                logger.debug(
                    "You are casting param %s to fp16, which"
                    " will cause loss of accuracy. This is required for"
                    " spyre cards that don't support bf16. You can ignore"
                    " this warning if this is intended.",
                    name,
                )
                param.data = param.data.to(dtype=torch.float16)

    def _cast_to_f32(self):
        """Cast model parameters to f32."""
        for name, param in self.fms_model.named_parameters():
            logger.debug(
                "Casting param %s to fp32. This is required"
                " for attention implementations that only support"
                " full precision.",
                name,
            )
            param.data = param.data.to(dtype=torch.float32)

    @staticmethod
    def resolve_hf_config(vllm_config: VllmConfig):
        """Ensure that we convert to the correctly typed subclass of PretrainedConfig;
        this is largely done to handle external config formats consistently, such
        as Mistral.

        NOTE: We should be careful about considering the implications of the external
        format with respect to the model arch used in upline vLLM (outside of Spyre),
        because in some cases, i.e., Mistral3, we map to a different vLLM class;
        this also may have implications for the preprocessing used by the input processor
        that runs in the frontend process, so we need to be sure that things are handled
        correctly on the mm utils side as well.
        """
        model_config: ModelConfig = vllm_config.model_config
        hf_config: PretrainedConfig = model_config.hf_config
        is_hf_format = model_config.config_format == "hf"
        vllm_arch = model_config.architecture

        # For multimodal mistral, passing auto / mistral format may result in passing a
        # PretrainedConfig that should be cast to Mistral3; In this case, we expect to
        # have arch PixtralForConditionalGeneration (mistral format for mistral3), as
        # opposed to Mistral3ForConditionalGeneration (HF format).
        # Ref: https://github.com/vllm-project/vllm/blob/v0.15.0/docs/models/supported_models.md
        if vllm_arch == "PixtralForConditionalGeneration" and type(hf_config) is PretrainedConfig:
            if is_hf_format:
                raise AssertionError(
                    "Mistral3 config format should not be hf with PixtralForConditionalGeneration"
                )
            if not hasattr(hf_config, "text_config") or not hasattr(hf_config, "vision_config"):
                raise AttributeError(
                    "Mistral3 config to be converted must have text/vision subconfigs"
                )

            logger.info("Converting from Mistral -> HF Config format for Mistral3")

            # Clobber the text / language model_types with the mistral/pixtral key, which
            # are currently the only models that we support in FMS with mistral3 at the moment,
            # and the values we would expect if we passed the config in HF format.
            config_dict = hf_config.to_dict()
            config_dict["text_config"]["model_type"] = "mistral"
            config_dict["vision_config"]["model_type"] = "pixtral"
            config_dict["model_type"] = "mistral3"
            return Mistral3Config(**config_dict)

        return hf_config

    def set_past_key_value_states(self, num_blocks) -> None:
        # List[layers] of Tuple[k,v] of
        # Tensor[num_blocks, block_size, num_kv_heads, head_dim]
        if not self.model_config.quantization:
            self.past_key_value_states = [
                (
                    torch.zeros(
                        num_blocks,
                        self.kv_cache_specs["block_size"],
                        self.kv_cache_specs["num_kv_heads"],
                        self.kv_cache_specs["head_dim"],
                        dtype=self.dtype,
                    ),
                    torch.zeros(
                        num_blocks,
                        self.kv_cache_specs["block_size"],
                        self.kv_cache_specs["num_kv_heads"],
                        self.kv_cache_specs["head_dim"],
                        dtype=self.dtype,
                    ),
                )
                for _ in range(self.kv_cache_specs["num_layers"])
            ]
        else:
            from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor

            batch_size = max(2, self.scheduler_config.max_num_seqs)
            self.past_key_value_states = [
                (
                    ScaledTensor(
                        torch.zeros(
                            num_blocks,
                            self.kv_cache_specs["block_size"],
                            self.kv_cache_specs["num_kv_heads"],
                            self.kv_cache_specs["head_dim"],
                            dtype=self.dtype,
                        ),
                        scale=torch.tensor([1.0] * batch_size, dtype=torch.float32),
                        scaled=False,
                    ),
                    ScaledTensor(
                        torch.zeros(
                            num_blocks,
                            self.kv_cache_specs["block_size"],
                            self.kv_cache_specs["num_kv_heads"],
                            self.kv_cache_specs["head_dim"],
                            dtype=self.dtype,
                        ),
                        scale=torch.tensor([1.0] * batch_size, dtype=torch.float32),
                        scaled=False,
                    ),
                )
                for _ in range(self.kv_cache_specs["num_layers"])
            ]

    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:
        forward_context = get_forward_context()

        attn_metadata = cast(SpyreAttentionMetadata, forward_context.attn_metadata)
        assert attn_metadata is not None
        # FMS does not eagerly register the paged attention algorithm, we must import it here
        import fms.utils.spyre.paged  # noqa # pylint: disable=unused-import

        if self.is_fp8_model:
            # set scale for kv_cache
            self._set_scale_for_fp8(attn_metadata)

            # Adjust decode for bs=1 if needed
            input_ids_or_embeds, positions, attn_metadata = self._adjust_input_for_fp8(
                input_ids=input_ids_or_embeds, position_ids=positions, attn_metadata=attn_metadata
            )

        # Run the model
        output = self.fms_model(
            input_ids_or_embeds,
            position_ids=positions,
            mask=masks,
            past_key_value_states=self.past_key_value_states,
            use_cache=True,
            last_n_tokens=SpyrePlatform.get_block_size() if is_prompt else 1,
            current_tkv_mask=attn_metadata.current_tkv_mask,
            left_padded_prompt_mask=attn_metadata.left_padded_prompt_mask,
            block_table=attn_metadata.block_table,
            slot_mapping=attn_metadata.slot_mapping,
            attn_name=self.attention_name,
        )

        # The second item in the output tuple is the KV cache.
        # However, on spyre these are ghost tensors- the data in these tensors does not reflect the
        # actual kv cache data on the device. They exist only for proper compilation, so we don't
        # waste any time assigning these tensors back to anything.
        logits, kv_cache = output
        if not self.on_spyre:
            self.past_key_value_states = kv_cache

        if is_prompt:
            # assert that indeed received the last block of logits
            assert logits.shape[1] == SpyrePlatform.get_block_size()

        if self.is_fp8_model:
            # If we weren't using static scaling, we would need to update the scales here.
            # This adjustment is for the extra padding to batch size 2 required by pytorch<=2.7
            logits = self._adjust_output_for_fp8(logits, attn_metadata)

        if is_prompt and self.n_pads_right > 0:
            # get last token before the right padding
            logits = logits[self.indices, -self.n_pads_right - 1, :]
        else:
            # just take last token if no right padding
            logits = logits[self.indices, -1, :]

        return logits

    def get_maybe_mm_embeddings(self, input_ids, mm_features, is_decode):
        """If the model is multimodal, get the (maybe) multimodal embeddings.
        If it isn't, return None, since we only use embeddings for multimodal.

        In the case of prefill / decode; we should only have mm features in
        prefill, because by that point, the multimodal data will already be
        merged into the embeddings to be cached. As such we explicitly explode
        if mm_features are passed in decode, because it's likely a mistake.

        NOTE: generally is_decode will set iteration > 0 in FMS; failing to do
        this will cause it to return the raw input id and try to embed in the
        forward call, which may break on AIU due to misalignment with prefill's
        embedding call.
        """
        if not self.is_multimodal or self.mm_model_utils is None:
            # The model is likely implemented incorrectly or not initialized,
            # or we are passing multimodal features to a model that should not
            # take them.
            if mm_features:
                raise ValueError("mm_features were provided, but model is not multimodal!")
            # We do not use embeddings for models that aren't multimodal.
            return None

        # Delegate to this model architecture's multimodal helpers to
        # get the (potentially) multimodal embeddings from the FMS model.
        fms_model = self.fms_model
        return self.mm_model_utils.get_maybe_mm_embeddings(
            fms_model,
            input_ids,
            mm_features,
            is_decode,
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_mask_dtype(self) -> torch.dtype:
        return self._mask_dtype

    def _set_scale_for_fp8(self, attn_metadata: SpyreAttentionMetadata):
        for _, (k, v) in enumerate(self.past_key_value_states):
            # Static scaling: We always set the scale to 1.0
            # There is probably an optimization here to not rebuild these [1.0] tensors
            k = cast(ScaledTensor, k)
            v = cast(ScaledTensor, v)
            if attn_metadata.is_prefill:
                k._scale = torch.ones(1, dtype=torch.float32)
                v._scale = torch.ones(1, dtype=torch.float32)
            elif len(self.indices) == 1:
                k._scale = torch.ones(2, dtype=torch.float32)
                v._scale = torch.ones(2, dtype=torch.float32)
            else:
                k._scale = torch.ones(len(self.indices), dtype=torch.float32)
                v._scale = torch.ones(len(self.indices), dtype=torch.float32)

            k._scaled = True
            v._scaled = True

    def get_dtype(self) -> torch.dtype:
        # Get the model's data type
        # This should be:
        # FP32 for un-quantized models on cpu
        # FP16 for un-quantized models on spyre
        # FP8 (float8_e4m3fn) for quantized models
        # (only fp8 quantization is supported)
        if self.model_config.quantization:
            return torch.float8_e4m3fn
        else:
            if envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND in BACKEND_LIST:
                return torch.float16
            else:
                return torch.float32

    # TODO: this is not the best place to do. But we expect this to
    # be temporary and here should be easy to remove later
    def _adjust_input_for_fp8(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
    ):
        # NOTE: We only need to adjust the inputs for decode with
        # batch_size=2
        if attn_metadata.is_prefill or input_ids.shape[0] > 1:
            return input_ids, position_ids, attn_metadata

        input_ids = input_ids.repeat(2, 1)
        position_ids = position_ids.repeat(2, 1)
        attn_metadata = SpyreAttentionMetadata(
            slot_mapping=attn_metadata.slot_mapping.repeat(2, 1),
            current_tkv_mask=attn_metadata.current_tkv_mask.repeat(2),
            left_padded_prompt_mask=attn_metadata.left_padded_prompt_mask.repeat(2),
            block_table=attn_metadata.block_table.repeat(2, 1),
            is_prefill=attn_metadata.is_prefill,
        )
        return input_ids, position_ids, attn_metadata

    def _adjust_output_for_fp8(self, logits: torch.Tensor, attn_metadata: SpyreAttentionMetadata):
        if attn_metadata.is_prefill or len(self.indices) > 1:
            # skip for prefill or decode for bs>1
            return logits

        return logits[0].unsqueeze(0)
