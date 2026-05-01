"""
(Non e2e) tests related to Llava Next (Granite Vision); these tests
primarily verify the correctness of some of the helper utils, especially
with respect to the creation of warmup features.
"""

import copy

import pytest
import torch
from fms.models.llava_next import LlavaNextConfig
from fms.models.hf.config_utils.param_builders import build_llava_next_params
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from vllm.multimodal.inputs import MultiModalFeatureSpec

import sendnn_inference.multimodal as spyre_mm
from tests.spyre_util import REFERENCE_MODELS

GVISION_MODEL = REFERENCE_MODELS["ibm-granite/granite-vision-3.2-2b"]
# Marks all tests in this file as multimodal and CPU to match
# multimodal wf; tests in this file should be very fast.
pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# NOTE: --forked forks after module scoped fixtures
@pytest.fixture(scope="module")
def hf_config():
    """Get a transformers config for granite vision."""
    return AutoConfig.from_pretrained(
        GVISION_MODEL.name,
        revision=GVISION_MODEL.revision,
    )


@pytest.fixture(scope="module")
def fms_config(hf_config):
    """Get the FMS config corresponding to the above."""
    config_params = build_llava_next_params(hf_config)
    config_params["text_config"].head_dim = 128
    return LlavaNextConfig(**config_params)


@pytest.fixture(scope="module")
def llava_next_mm_utils(fms_config, hf_config):
    return spyre_mm.maybe_get_mm_utils(
        model_path=GVISION_MODEL.name,
        fms_config=fms_config,
        hf_config=hf_config,
    )


def test_loads_correct_mm_utils(llava_next_mm_utils):
    """Ensure that we map the config to the right mm utils subclass."""
    assert llava_next_mm_utils is not None
    assert isinstance(llava_next_mm_utils, spyre_mm.LlavaNextMMUtils)


def test_config_validation(fms_config, hf_config):
    """Ensure that init fails if llava next is initialized for
    a non-granite LLM (currently only support granite vision).
    """
    non_granite_cfg = copy.deepcopy(hf_config)
    non_granite_cfg.text_config.model_type = "not granite"
    with pytest.raises(TypeError):
        spyre_mm.LlavaNextMMUtils._validate_configs(
            fms_config=fms_config,
            hf_config=non_granite_cfg,
        )


### Tests for inspecting the correctness of the warmup shapes
def test_warmup_embed_types_and_shape(llava_next_mm_utils):
    """Ensure that the types and dimensions for the embeddings are consistent
    with the input IDs. Note that currently we pass input IDs all the way
    through, so these should exist for sanity even though the model processes
    the embeddings.
    """
    warmup_inputs = llava_next_mm_utils.get_warmup_inputs(req_count=1)
    warmup_toks = torch.Tensor(warmup_inputs.input_ids)[0]
    warmup_embeds_tensor = warmup_inputs.input_embeds[0]

    assert isinstance(warmup_toks, torch.Tensor)
    # Ensure embeddings and tokens have the same dims except the embed dim
    assert warmup_toks.shape == warmup_embeds_tensor.shape[:-1]
    # Check the embedding shape is consistent with the text subconfig
    assert warmup_embeds_tensor.shape[-1] == llava_next_mm_utils.hf_config.text_config.hidden_size
    assert isinstance(warmup_embeds_tensor, torch.Tensor)


def test_warmup_mm_features_types(llava_next_mm_utils):
    """Check to ensure the mm features correspond to one image."""
    warmup_inputs = llava_next_mm_utils.get_warmup_inputs(req_count=1)
    warmup_mm_features = warmup_inputs.mm_features

    # Multimodal features should be a list of (one) multimodal feature spec,
    # since we warm this model up with features pertaining to one image.
    assert isinstance(warmup_mm_features, list)
    assert len(warmup_mm_features) == 1
    assert all(isinstance(spec, MultiModalFeatureSpec) for spec in warmup_mm_features)


def test_warmup_shape_alignment(llava_next_mm_utils):
    """Compare the alignment between the multimodal feature spec contents
    and the input embeddings etc; this ensures that the expanded image tokens
    actually correctly align with the embeddings and mm feature spec to prevent
    alignment issues when we merge the multimodal embeddings in FMS.
    """
    warmup_inputs = llava_next_mm_utils.get_warmup_inputs(req_count=1)
    warmup_toks = torch.Tensor(warmup_inputs.input_ids)
    warmup_mm_features = warmup_inputs.mm_features[0]

    # Get the total number of expanded image tokens in the inputs
    image_token_id = llava_next_mm_utils.get_multimodal_token_id()
    num_expanded_mm_ids = torch.sum(warmup_toks == image_token_id).item()

    # Only have one input image, so one image, so offsets should match directly
    im_start = warmup_mm_features.mm_position.length
    im_end = warmup_mm_features.mm_position.offset
    assert num_expanded_mm_ids == (im_start - im_end)


def test_warmup_feature_correctness(llava_next_mm_utils):
    """Ensure that the expanded image token count is actually
    correct with respect to the input image size by reprocessing
    it with transformers and comparing the result.

    NOTE: depending on the implementation, this may be redundant,
    as the llava next processor essentially does this internally.
    However, if other multimodal models are implemented, this is
    is an important thing to verify.
    """
    image_token_id = llava_next_mm_utils.get_multimodal_token_id()

    warmup_inputs = llava_next_mm_utils.get_warmup_inputs(req_count=1)
    warmup_toks = torch.Tensor(warmup_inputs.input_ids)
    warmup_mm_features = warmup_inputs.mm_features[0]

    num_expanded_mm_ids = torch.sum(warmup_toks == image_token_id).item()

    processor = AutoProcessor.from_pretrained(GVISION_MODEL.name, revision=GVISION_MODEL.revision)
    # Create a random PIL Image that matches the size of the hardcoded
    # inputs and run it through the processor to check the feature sizes.
    image_dims = warmup_mm_features.data["image_sizes"].data

    # NOTE: Shape is width x height for PIL, but it's height x width in pytorch,
    # so we need to flip dims here to ensure alignment with the torch tensors.
    dummy_img = Image.new("RGB", image_dims.tolist()[::-1])

    img_tok = processor.decode(image_token_id)
    preproc_res = processor(
        images=dummy_img,
        text=img_tok,
        return_tensors="pt",
    )
    actual_expanded_mm_ids = torch.sum(preproc_res.input_ids == image_token_id).item()

    # Check that the hardcoded number of image toks matches the processor result
    assert num_expanded_mm_ids == actual_expanded_mm_ids

    # Check that after squeezing, the 4D pixel vals are the same shape
    pixel_vals = preproc_res.pixel_values.squeeze(0)
    assert pixel_vals.shape == warmup_mm_features.data["pixel_values"].data.shape

    # Check that the image shapes are also correct; note the dim flip earlier,
    # this is generally important because an m x n image does not necessarily
    # encode to the same number of tokens (or even tiles) as an n x m.
    # For now, m == n in the warmup image, so this is mostly to make the
    # test less brittle.
    assert bool(torch.all(preproc_res.image_sizes == image_dims))
