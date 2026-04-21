import transformers

from sendnn_inference.multimodal.mm_mappings import LlavaNextMMUtils, Mistral3MMUtils, MMUtilsBase

# Maps transformers classes to the corresponding utils
MM_HF_CFG_REGISTRY = {
    transformers.LlavaNextConfig: LlavaNextMMUtils,
    transformers.Mistral3Config: Mistral3MMUtils,
}


def get_mm_specific_load_overrides(hf_config: transformers.PretrainedConfig):
    # Ensure the model is multimodal, otherwise we have no overrides
    cfg_type = type(hf_config)
    if cfg_type not in MM_HF_CFG_REGISTRY:
        return {}
    return MM_HF_CFG_REGISTRY[cfg_type].get_mm_specific_load_overrides(hf_config)


def maybe_get_mm_utils(model_path, fms_config, hf_config) -> MMUtilsBase | None:
    """Create an instance of the corresponding multimodal model's utils
    if one exists; if it doesn't, the model is not multimodal.
    """
    # Note: mistral doesn't come up with transformers.Mistral3Config and needs to be
    # typecasted separately. So for now, we are just checking with its detected model_type,
    # which is pixtral

    if type(hf_config) in MM_HF_CFG_REGISTRY:
        util_cls = MM_HF_CFG_REGISTRY[type(hf_config)]
        return util_cls(
            model_path=model_path,
            fms_config=fms_config,
            hf_config=hf_config,
        )

    return None  # Not a multimodal model
