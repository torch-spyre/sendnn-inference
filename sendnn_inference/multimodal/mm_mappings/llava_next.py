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
        mm_device: str,
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

                pixel_values = mm_spec["pixel_values"].data
                # Place pixel_values on the same device/dtype as the
                # vision_tower so the encoder forward can run on NNPA when the
                # vision_tower weights ended up on nnpa (CPU otherwise).
                mm_dtype = envs_spyre.SENDNN_INFERENCE_CPU_MM_DTYPE
                if pixel_values.device.type != mm_device or pixel_values.dtype != mm_dtype:
                    pixel_values = pixel_values.to(device=mm_device, dtype=mm_dtype)
                fms_kwargs["pixel_values"] = pixel_values

                image_sizes = mm_spec["image_sizes"].data

                # Careful about this; if it's 1D, we'll a tensor of shape
                # [x, y], which will break in a weird way in image packing,
                # since it assumes it's 2D and will get sad about getting
                # an int instead of an iterable
                if image_sizes.ndim == 1:
                    image_sizes = image_sizes.unsqueeze(0)
                # image_sizes is an integer index tensor; keep it on CPU
                # (NNPA dispatch for int tensors would just fall back anyway).
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

    @staticmethod
    def get_mm_embeddings_batch(
        fms_model,
        batch_input_ids: list,
        batch_mm_features: list,
        mm_device: str,
    ) -> list:
        """Run vision encoder once for all requests and return per-request
        embeddings each of shape [1, seq_len_i, emb_dim].

        pixel_values from all requests are stacked into a 5-D tensor
        [N, max_patches, C, H, W] so the SiGLIP vision tower processes all of
        them in one forward pass.  Each request is assumed to have exactly one
        image (the existing single-image-per-request constraint).
        """
        mm_dtype = envs_spyre.SENDNN_INFERENCE_CPU_MM_DTYPE
        mm_spec_keys = ["pixel_values", "image_sizes"]

        all_pixel_values: list[torch.Tensor] = []
        all_image_sizes: list[torch.Tensor] = []

        for mm_features in batch_mm_features:
            assert len(mm_features) == 1, (
                "LlavaNext get_mm_embeddings_batch expects exactly one image per request"
            )
            mm_spec = mm_features[0].data
            if mm_spec is None or any(k not in mm_spec for k in mm_spec_keys):
                raise KeyError(f"Llava Next requires kwargs: {mm_spec_keys}")

            pixel_values = mm_spec["pixel_values"].data
            if pixel_values.device.type != mm_device or pixel_values.dtype != mm_dtype:
                pixel_values = pixel_values.to(device=mm_device, dtype=mm_dtype)
            # pixel_values is [num_patches, C, H, W] for one image
            all_pixel_values.append(pixel_values)

            image_sizes = mm_spec["image_sizes"].data
            if image_sizes.ndim == 1:
                image_sizes = image_sizes.unsqueeze(0)
            all_image_sizes.append(image_sizes)

        # Stack into [N, max_patches, C, H, W] — SiGLIP handles variable patch
        # counts via the 5-D branch in get_image_features.
        max_patches = max(pv.shape[0] for pv in all_pixel_values)
        n = len(all_pixel_values)
        stacked_pv = torch.zeros(
            n,
            max_patches,
            *all_pixel_values[0].shape[1:],
            dtype=all_pixel_values[0].dtype,
            device=all_pixel_values[0].device,
        )
        for i, pv in enumerate(all_pixel_values):
            stacked_pv[i, : pv.shape[0]] = pv

        stacked_image_sizes = torch.cat(all_image_sizes, dim=0)  # [N, 2]

        # Single vision-tower forward for all N images.
        image_features = fms_model.get_image_features(stacked_pv, stacked_image_sizes)
        packed_features = fms_model.pack_image_features(
            image_features,
            stacked_image_sizes,
            image_newline=fms_model.image_newline,
        )
        packed_features = packed_features.to(
            dtype=fms_model.language_model.base_model.embedding.weight.dtype
        )

        # Build per-request embeddings: embed tokens and insert image features.
        # The updated prepare_inputs_for_generation loops over batch rows, so we
        # invoke it with a padded batch.  Alternatively (and more efficiently),
        # we do the insertion directly here to avoid any padding overhead.
        results: list[torch.Tensor] = []
        offset = 0
        for req_idx, (input_ids_1d, mm_features) in enumerate(
            zip(batch_input_ids, batch_mm_features)
        ):
            input_ids_2d = input_ids_1d.unsqueeze(0)  # [1, seq_len]
            embeds = fms_model.language_model.base_model.embedding(input_ids_2d)

            image_positions = (input_ids_1d == fms_model.config.image_token_index).nonzero(
                as_tuple=True
            )[0]
            n_img_toks = image_positions.shape[0]
            embeds[0, image_positions] = packed_features[offset : offset + n_img_toks]
            offset += n_img_toks
            results.append(embeds)

        return results

    def get_multimodal_token_id(self) -> int:
        return self.hf_config.image_token_index
