import functools
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from fms.utils.config import ModelConfig
from transformers import AutoProcessor, PretrainedConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec


class MMWarmupInputs(NamedTuple):
    """Wrapper for multimodal model warmup inputs,
    used for continuous batching."""

    input_ids: list[list[int]]
    input_embeds: list[torch.Tensor]
    mm_features: list[MultiModalFeatureSpec]


class MMUtilsBase(ABC):
    """
    Helpers utilities that are typically model architecture specific,
    and may be needed for properly integrating multimodal models from
    FMS.

    Note that by convention, to avoid confusion, we use fms_config to
    refer to the FMS model config & hf_config to refer to the transformers
    config (i.e., avoid ambiguous terms like model_config for readability).
    """

    def __init__(self, model_path: str, fms_config: ModelConfig, hf_config: PretrainedConfig):
        self._validate_configs(fms_config, hf_config)
        self.fms_config = fms_config
        self.hf_config = hf_config
        self.model_path = model_path

    @functools.cached_property
    def hf_processor(self):
        """Get the Transformers processor, but only if we need it."""
        return AutoProcessor.from_pretrained(self.model_path)

    @staticmethod
    def _validate_configs(fms_config: ModelConfig, hf_config: PretrainedConfig):
        """Ensure that configs are properly typed. Additional validation, e.g.,
        validating subconfig attrs should generally be done within subclasses.
        """
        if not isinstance(fms_config, ModelConfig):
            raise TypeError(
                "Provided fms_config is of type %s, not an FMS ModelConfig", type(fms_config)
            )

        if not isinstance(hf_config, PretrainedConfig):
            raise TypeError(
                "Provided hf_config is of type %s, not a PretrainedConfig", type(fms_config)
            )

    def resolve_multimodal_vocab_size(self) -> int:
        """Determine the vocabulary size of the underlying LLM, which
        is wrapped by a composite multimodal model.
        """
        # Try to look for the src_vocab_size in the sub text_config.
        # This will work for models like granite vision, but ultimately
        # depends on the wrapping vLM and underlying LLM.
        if text_config := getattr(self.fms_config, "text_config", None):
            if vocab_sz := getattr(text_config, "src_vocab_size", None):
                return vocab_sz
            raise ValueError("Provided FMS config has a text_config, but no src_vocab_size!")
        raise ValueError("Provided FMS config has no text config!")

    def unwrap_mm_kv_cache_opts(self):
        """Unwrap options to be passed for the kv cache from the underlying
        text configs and return the resulting dictionary, which is used to
        .update() the common kv cache opts that don't need unwrapping.
        """
        return {}

    @staticmethod
    def get_mm_specific_load_overrides(hf_config: PretrainedConfig):
        """Get any overrides needed for fixing compile with current multimodal
        models when calling from fms.models.get_model(); this should largely
        remain as static since it should only be used when initialized the FMS
        model, which will give us the FMS config.
        """
        return {}

    @staticmethod
    @abstractmethod
    def get_maybe_mm_embeddings(
        fms_model: torch.nn.Module,
        input_ids: torch.Tensor,
        mm_features: list[MultiModalFeatureSpec],
        is_decode: bool,
    ) -> torch.Tensor:
        """Get the (potentially) multimodal embeddings for this model
        architecture. Produced tensors should be of shape
        [bsz, seq_len, emb_dim].
        """
        pass

    @abstractmethod
    def get_warmup_inputs(self, req_count: int) -> MMWarmupInputs:
        pass

    @abstractmethod
    def get_multimodal_token_id(self) -> int:
        pass
