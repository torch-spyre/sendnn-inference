import torch
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


class Mistral3MMUtils(MMUtilsBase):
    image_token = "[IMG]"

    @staticmethod
    def _validate_configs(fms_config: ModelConfig, hf_config: PretrainedConfig):
        """Ensure that configs are properly typed. Additional validation, e.g.,
        validating subconfig attrs should generally be done within subclasses.
        """
        MMUtilsBase._validate_configs(fms_config, hf_config)
        if hf_config.model_type != "mistral3" or hf_config.text_config.model_type != "mistral":
            # HF config maps mistral3 model_type to pixtral
            raise TypeError("mistral3 currently only supports mistral LLMs!")

    def unwrap_mm_kv_cache_opts(self):
        """Unwrap options to be passed for the kv cache from the underlying
        text configs and return the resulting dictionary, which is used to
        .update() the common kv cache opts that don't need unwrapping.
        """
        kv_cache_specs = {}
        # NOTE: this is mistral LLM specific, since the only mistral3
        # variant supported in FMS is currently pixtral.
        kv_cache_specs["num_layers"] = self.hf_config.text_config.num_hidden_layers
        kv_cache_specs["head_dim"] = getattr(
            self.fms_config.text_config, "head_dim", self.hf_config.text_config.head_dim
        )
        return kv_cache_specs

    @staticmethod
    def get_maybe_mm_embeddings(
        fms_model: torch.nn.Module,
        input_ids: torch.Tensor,
        mm_features: list[MultiModalFeatureSpec],
        is_decode: bool,
    ) -> torch.Tensor:
        """Get the text or multimodal embeddings for mistral3 using
        the (potentially compiled) FMS model.
        """
        fms_kwargs = {"use_cache": True}

        # Only merge multimodal features in prefill; nothing mm in decode
        if mm_features:
            # Looks for ["pixel_values", "image_sizes"] in mm_features
            if len(mm_features) != 1:
                raise ValueError("Currently we assume we only embed one mm request at a time")
            mm_spec = mm_features[0].data

            # when using config and tokenizer are set to `mistral` we don't get
            # pixel_values in mm_spec. So we are mapping these back here
            if isinstance(mm_spec, MultiModalKwargsItem) and "images" in mm_spec:
                mm_spec["pixel_values"] = mm_spec.pop("images")

            if mm_spec is not None:
                if "pixel_values" not in mm_spec:
                    raise KeyError("Mistral3 requires pixel_values")

                pixel_values = mm_spec["pixel_values"].data
                # FMS vision tower expects pixel_values with batch dimension
                # If squeezed during spec building, add it back
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                fms_kwargs["pixel_values"] = pixel_values

                if "image_sizes" in mm_spec:
                    # Use the processor's image_sizes which tracks the logical image dimensions
                    # This is used by the projector to correctly split/merge patches
                    image_sizes_tensor = mm_spec["image_sizes"].data
                    if image_sizes_tensor.ndim == 1:
                        # Single image: convert to list of tuples
                        image_sizes = [(image_sizes_tensor[0].item(), image_sizes_tensor[1].item())]
                    else:
                        # Multiple images
                        image_sizes = [(h.item(), w.item()) for h, w in image_sizes_tensor]
                else:
                    # Mistral image input in vLLM doesn't contain image_sizes as attribute, so we
                    # are calculating based on pixel_values
                    # Ref: https://github.com/vllm-project/vllm/blob/f97ca671766c5201404e9fc812e35bf2c4e95a01/vllm/model_executor/models/mistral3.py#L516C9-L518C10
                    image_sizes = [(img.shape[-2], img.shape[-1]) for img in pixel_values]

                fms_kwargs["image_sizes"] = image_sizes

        # The value of iteration does not matter for decode as long as it's > 0
        input_embeds, _ = fms_model.prepare_inputs_for_generation(
            iteration=0 if not is_decode else 1, input_ids=input_ids, kwargs=fms_kwargs
        )  # ty: ignore[call-non-callable]
        return input_embeds

    def get_warmup_inputs(self, req_count: int) -> MMWarmupInputs:
        """Generate input for warmup using using dummy image."""

        # Get vision config parameters
        patch_size = self.hf_config.vision_config.patch_size
        spatial_merge_size = getattr(self.hf_config, "spatial_merge_size", 2)
        image_token_id = self.hf_config.image_token_index
        emb_dim = self.hf_config.text_config.hidden_size

        # Warmup with minimal nontrivial case (4x4 patches)
        # Note: spatial_merge_size for mistral is 2, which means after merging,
        # a 4x4 patch grid becomes 2x2 = 4 image tokens
        # In FMS currently, we do squeeze(0) on image features in
        # _get_image_features function before splitting, which means, if we only have 1
        # patch, and 1st dim is is 1, we get incorrect dimension of image_features
        side_dim = patch_size * 4
        num_patches_per_side = 4
        num_merged_patches_per_side = num_patches_per_side // spatial_merge_size  # 2
        num_image_tokens = num_merged_patches_per_side * num_merged_patches_per_side  # 4

        # Create input_ids using image tokens
        warmup_input_ids = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)

        # Create random embeddings
        warmup_embeds = torch.rand((num_image_tokens, emb_dim))

        # Create dummy pixel_values: normalized float16 tensor (legal format for vision encoder)
        dummy_pixel_values = torch.rand((3, side_dim, side_dim), dtype=torch.float16)

        # Create image_sizes: logical dimensions of the image
        dummy_image_sizes = torch.tensor([side_dim, side_dim], dtype=torch.long)

        # Build multimodal features spec
        mm_position = PlaceholderRange(offset=0, length=num_image_tokens)
        mm_data = {
            "pixel_values": dummy_pixel_values,
            "image_sizes": dummy_image_sizes,
        }
        mm_fields = MultiModalKwargsItem(
            {
                mm_key: MultiModalFieldElem(data=mm_data, field=MultiModalBatchedField())
                for mm_key, mm_data in mm_data.items()
            }
        )
        warmup_mm_features = [
            MultiModalFeatureSpec(
                data=mm_fields,
                modality="image",
                identifier="MM-warmup-mistral3",
                mm_position=mm_position,
            )
        ]

        return MMWarmupInputs(
            input_ids=[warmup_input_ids.tolist()] * req_count,
            input_embeds=[warmup_embeds] * req_count,
            mm_features=warmup_mm_features,
        )

    def get_multimodal_token_id(self) -> int:
        return self.hf_config.image_token_index
