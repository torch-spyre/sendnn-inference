import torch
from fms.utils import serialization
from fms.utils.config import ModelConfig
from transformers import PretrainedConfig
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

from sendnn_inference.multimodal.mm_mappings import MMUtilsBase, MMWarmupInputs

# Extend the adapter as part of the head dim fix; this is needed to
# load 2b models correctly, but we do it here since this class is
# currently initialized only once and the adapter extension does not
# seem to be idempotent.
#
# NOTE: If this is made idempotent, we can move this into
# get_mm_specific_load_overrides(), since it's needed to load.
serialization.extend_adapter("llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"])


class LlavaNextMMUtils(MMUtilsBase):
    @staticmethod
    def _validate_configs(fms_config: ModelConfig, hf_config: PretrainedConfig):
        """Ensure that configs are properly typed. Additional validation, e.g.,
        validating subconfig attrs should generally be done within subclasses.
        """
        MMUtilsBase._validate_configs(fms_config, hf_config)
        if hf_config.model_type != "llava_next" or hf_config.text_config.model_type != "granite":
            raise TypeError("llava next currently only supports granite LLMs!")

    def unwrap_mm_kv_cache_opts(self):
        """Unwrap options to be passed for the kv cache from the underlying
        text configs and return the resulting dictionary, which is used to
        .update() the common kv cache opts that don't need unwrapping.
        """
        kv_cache_specs = {}
        # NOTE: this is granite LLM specific, since the only llava next
        # variant supported in FMS is currently granite vision.
        kv_cache_specs["num_layers"] = self.hf_config.text_config.num_hidden_layers
        kv_cache_specs["head_dim"] = getattr(
            self.fms_config.text_config,
            "head_dim",
            self.hf_config.text_config.hidden_size
            // self.hf_config.text_config.num_attention_heads,
        )
        return kv_cache_specs

    @staticmethod
    def get_mm_specific_load_overrides(hf_config: PretrainedConfig):
        """Get any overrides needed for initializing the FMS model from the
        transformers config. For this model, we need to fix the head_dim, which
        currently surfaces as a problem for all 2b variants of granite 3.x LLMs
        when running through FMS.

        TODO: If additional variants of granite vision are added, or broader
        llava next support is added in FMS, handle it properly here.
        """
        return {
            "override_hf_pretrained_config": True,
            "text_config": {"head_dim": 128},
        }

    @staticmethod
    def get_maybe_mm_embeddings(
        fms_model: torch.nn.Module,
        input_ids: torch.Tensor,
        mm_features: list[MultiModalFeatureSpec],
        is_decode: bool,
    ) -> torch.Tensor:
        """Get the text or multimodal embeddings for Llava Next using
        the (potentially compiled) FMS model.
        """
        fms_kwargs = {"use_cache": True}
        mm_spec_keys = ["pixel_values", "image_sizes"]

        # Only merge multimodal features in prefill; nothing mm in decode
        if mm_features:
            assert not is_decode  # We never pass features in decode
            if len(mm_features) != 1:
                raise ValueError("Currently we assume we only embed one mm request at a time")
            mm_spec = mm_features[0].data
            if mm_spec is not None:
                # NOTE: This should be pretty safe as it's dependent on the
                # vLLM/HF processor objects, but we check it anyway to be safe
                # for now, since transformers 5.0 is just around the corner.
                if any(k not in mm_spec for k in mm_spec_keys):
                    raise KeyError(f"Llava Next requires kwargs: {mm_spec_keys}")

                fms_kwargs["pixel_values"] = mm_spec["pixel_values"].data
                image_sizes = mm_spec["image_sizes"].data

                # Careful about this; if it's 1D, we'll a tensor of shape
                # [x, y], which will break in a weird way in image packing,
                # since it assumes it's 2D and will get sad about getting
                # an int instead of an iterable
                if image_sizes.ndim == 1:
                    image_sizes = image_sizes.unsqueeze(0)
                fms_kwargs["image_sizes"] = image_sizes

        # The value of iteration does not matter for decode as long as it's > 0
        input_embeds, _ = fms_model.prepare_inputs_for_generation(
            iteration=0 if not is_decode else 1, input_ids=input_ids, kwargs=fms_kwargs
        )  # ty: ignore[call-non-callable]
        return input_embeds

    def get_warmup_inputs(self, req_count: int) -> MMWarmupInputs:
        """Get the inputs to the huggingface processor to create the warmup
        features or feature shapes.
        """
        # Warmup text is just an image token
        dummy_tokens = [self.hf_processor.decode(self.get_multimodal_token_id())]

        # number of image tokens only depends on shape;
        # using a smaller image here uses less context.
        tile_size = self.hf_config.vision_config.image_size
        side_dim = tile_size // 2
        dummy_img = torch.zeros((3, side_dim, side_dim), dtype=torch.uint8)

        proc_res = self.hf_processor(
            text=dummy_tokens,
            images=dummy_img,
            return_tensors="pt",
        )

        seq_len = proc_res.input_ids.shape[-1]
        # Get the input tokens and embeddings; currently embeddings are used,
        # but tokens are still required for the interfaces to be happy.
        warmup_input_ids = proc_res.input_ids.squeeze(0)
        emb_dim = self.hf_config.text_config.hidden_size
        warmup_embeds = torch.rand((seq_len, emb_dim))
        # Get the multimodal features spec
        warmup_mm_features = LlavaNextMMUtils._build_multimodal_spec(proc_res)

        return MMWarmupInputs(
            input_ids=[warmup_input_ids.tolist()] * req_count,
            input_embeds=[warmup_embeds] * req_count,
            mm_features=warmup_mm_features,
        )

    @staticmethod
    def _build_multimodal_spec(proc_res):
        """Given output of the processor on warmup data, build MM features"""

        # Squeeze down batch dim here; all token inputs are image tokens
        num_img_toks = proc_res.input_ids.shape[-1]

        # Multimodal features / feature spec
        mm_position = PlaceholderRange(offset=0, length=num_img_toks)
        mm_data = {
            "pixel_values": proc_res.pixel_values.squeeze(axis=0),
            "image_sizes": proc_res.image_sizes.squeeze(axis=0),
        }
        mm_fields = MultiModalKwargsItem(
            {
                mm_key: MultiModalFieldElem(data=mm_data, field=MultiModalBatchedField())
                for mm_key, mm_data in mm_data.items()
            }
        )

        return [
            MultiModalFeatureSpec(
                data=mm_fields,
                modality="image",
                identifier="MM-warmup-llava-next",
                mm_position=mm_position,
            )
        ]

    def get_multimodal_token_id(self) -> int:
        return self.hf_config.image_token_index
