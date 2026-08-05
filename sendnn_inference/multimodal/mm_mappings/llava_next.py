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

import sendnn_inference.envs as envs_spyre
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
        transformers config. For granite-vision-3.2-2b in HF format, the
        text_config doesn't declare `head_dim` explicitly, and the default
        fallback (`hidden_size / num_attention_heads` = 2048/32 = 64) is
        wrong for the FMS Granite implementation, which expects 128. We
        patch it in here.

        A config that already sets `head_dim` correctly (whether to 128 or
        any other legal value) needs no override — returning a partial
        `text_config` dict here with `override_hf_pretrained_config=True`
        would obliterate every other field FMS built from the HF config
        (emb_dim, nheads, etc.) because FMS's override merges are
        whole-key replacements, not deep merges.
        """
        text_cfg = hf_config.text_config
        explicit_head_dim = getattr(text_cfg, "head_dim", None)
        if explicit_head_dim is not None:
            # HF config knows its own head_dim — trust it.
            return {}

        implicit_head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        if implicit_head_dim == 128:
            # Implicit head_dim already matches what FMS wants; no override.
            return {}

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
        mm_device: str,
    ) -> torch.Tensor:
        """Get the text or multimodal embeddings for Llava Next using
        the (potentially compiled) FMS model.
        """
        fms_kwargs = {"use_cache": True}

        # Only merge multimodal features in prefill; nothing mm in decode
        if mm_features:
            assert not is_decode  # We never pass features in decode
            # Shared prep with encode_images (keeps the fused and decomposed
            # paths from drifting). image_sizes stays on CPU (integer index tensor).
            pixel_values, image_sizes = LlavaNextMMUtils._prepare_vision_inputs(
                mm_features, mm_device
            )
            fms_kwargs["pixel_values"] = pixel_values
            fms_kwargs["image_sizes"] = image_sizes

        # The value of iteration does not matter for decode as long as it's > 0
        input_embeds, _ = fms_model.prepare_inputs_for_generation(
            iteration=0 if not is_decode else 1, input_ids=input_ids, kwargs=fms_kwargs
        )  # ty: ignore[call-non-callable]
        return input_embeds

    @staticmethod
    def _prepare_vision_inputs(
        mm_features: list[MultiModalFeatureSpec], mm_device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and prepare ``(pixel_values, image_sizes)`` for the vision tower.

        Shared by ``get_maybe_mm_embeddings`` and ``encode_images`` so the two
        paths cannot drift. ``pixel_values`` is placed on the vision_tower's
        device/dtype; ``image_sizes`` is promoted to 2D (kept on CPU — it is an
        integer index tensor).
        """
        if len(mm_features) != 1:
            raise ValueError("Currently we assume we only embed one mm request at a time")
        mm_spec = mm_features[0].data
        mm_spec_keys = ["pixel_values", "image_sizes"]
        if mm_spec is None or any(k not in mm_spec for k in mm_spec_keys):
            raise KeyError(f"Llava Next requires kwargs: {mm_spec_keys}")

        pixel_values = mm_spec["pixel_values"].data
        mm_dtype = envs_spyre.SENDNN_INFERENCE_CPU_MM_DTYPE
        if pixel_values.device.type != mm_device or pixel_values.dtype != mm_dtype:
            pixel_values = pixel_values.to(device=mm_device, dtype=mm_dtype)

        image_sizes = mm_spec["image_sizes"].data
        if image_sizes.ndim == 1:
            image_sizes = image_sizes.unsqueeze(0)
        return pixel_values, image_sizes  # ty: ignore[invalid-return-type]

    @staticmethod
    def encode_images(
        fms_model: torch.nn.Module,
        mm_features: list[MultiModalFeatureSpec],
        mm_device: str,
    ) -> torch.Tensor:
        """Run the SiglipVision tower + projector for Llava Next and return the
        packed image features [num_image_tokens, emb_dim]."""
        pixel_values, image_sizes = LlavaNextMMUtils._prepare_vision_inputs(mm_features, mm_device)
        image_features = fms_model.get_image_features(pixel_values, image_sizes)  # ty: ignore[call-non-callable]
        return fms_model.pack_image_features(  # ty: ignore[call-non-callable]
            image_features, image_sizes, image_newline=fms_model.image_newline
        )

    @staticmethod
    def embed_text(fms_model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        return fms_model._get_text_embeddings(input_ids)  # ty: ignore[call-non-callable]

    @staticmethod
    def merge_embeddings(
        fms_model: torch.nn.Module,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        image_features = image_features.to(text_embeds.device, text_embeds.dtype)
        image_positions = (input_ids[0] == fms_model.config.image_token_index).nonzero(
            as_tuple=True
        )[0]
        text_embeds[0, image_positions] = image_features
        return text_embeds

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
