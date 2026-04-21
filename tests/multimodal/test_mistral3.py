"""
(Non e2e) tests related to Mistral3 (multimodal models); these tests
primarily verify the correctness of some of the helper utils, especially
with respect to the creation of warmup features.
"""

import copy

import pytest
import torch
from fms.models.mistral3 import Mistral3Config
from fms.models.hf.config_utils.param_builders import build_mistral3_params
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from vllm.multimodal.inputs import MultiModalFeatureSpec

import sendnn_inference.multimodal as spyre_mm
from tests.spyre_util import REFERENCE_MODELS

# MISTRAL3_MODEL = REFERENCE_MODELS["mistralai/Mistral-Small-3.2-24B-Instruct-2506"]
MISTRAL3_MODEL = REFERENCE_MODELS["mistralai/Mistral-Small-3.1-24B-Instruct-2503"]
# Marks all tests in this file as multimodal and CPU to match
# multimodal wf; tests in this file should be very fast.
pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# NOTE: --forked forks after module scoped fixtures
@pytest.fixture(scope="module")
def hf_config():
    """Get a transformers config for mistral3."""
    return AutoConfig.from_pretrained(
        MISTRAL3_MODEL.name,
        revision=MISTRAL3_MODEL.revision,
    )


@pytest.fixture(scope="module")
def fms_config(hf_config):
    """Get the FMS config corresponding to the above."""
    config_params = build_mistral3_params(hf_config)
    return Mistral3Config(**config_params)


@pytest.fixture(scope="module")
def mistral3_mm_utils(fms_config, hf_config):
    return spyre_mm.maybe_get_mm_utils(
        model_path=MISTRAL3_MODEL.name,
        fms_config=fms_config,
        hf_config=hf_config,
    )


def test_loads_correct_mm_utils(mistral3_mm_utils):
    """Ensure that we map the config to the right mm utils subclass."""
    assert mistral3_mm_utils is not None
    assert isinstance(mistral3_mm_utils, spyre_mm.Mistral3MMUtils)


def test_config_validation(fms_config, hf_config):
    """Ensure that init fails if mistral3 is initialized for
    a non-mistral LLM (i.e., FMS only supports mistral + pixtral).
    """
    non_mistral3_cfg = copy.deepcopy(hf_config)
    non_mistral3_cfg.text_config.model_type = "not mistral"
    with pytest.raises(TypeError):
        spyre_mm.Mistral3MMUtils._validate_configs(
            fms_config=fms_config,
            hf_config=non_mistral3_cfg,
        )


### Tests for inspecting the correctness of the warmup shapes
def test_warmup_embed_types_and_shape(mistral3_mm_utils):
    """Ensure that the types and dimensions for the embeddings are consistent
    with the input IDs. Note that currently we pass input IDs all the way
    through, so these should exist for sanity even though the model processes
    the embeddings.
    """
    warmup_inputs = mistral3_mm_utils.get_warmup_inputs(req_count=1)
    warmup_toks = torch.Tensor(warmup_inputs.input_ids)[0]
    warmup_embeds_tensor = warmup_inputs.input_embeds[0]

    assert isinstance(warmup_toks, torch.Tensor)
    # Ensure embeddings and tokens have the same dims except the embed dim
    assert warmup_toks.shape == warmup_embeds_tensor.shape[:-1]
    # Check the embedding shape is consistent with the text subconfig
    assert warmup_embeds_tensor.shape[-1] == mistral3_mm_utils.hf_config.text_config.hidden_size
    assert isinstance(warmup_embeds_tensor, torch.Tensor)


def test_warmup_mm_features_types(mistral3_mm_utils):
    """Check to ensure the mm features correspond to one image."""
    warmup_inputs = mistral3_mm_utils.get_warmup_inputs(req_count=1)
    warmup_mm_features = warmup_inputs.mm_features

    # Multimodal features should be a list of (one) multimodal feature spec,
    # since we warm this model up with features pertaining to one image.
    assert isinstance(warmup_mm_features, list)
    assert len(warmup_mm_features) == 1
    assert all(isinstance(spec, MultiModalFeatureSpec) for spec in warmup_mm_features)


def test_warmup_shape_alignment(mistral3_mm_utils):
    """Compare the alignment between the multimodal feature spec contents
    and the input embeddings etc; this ensures that the expanded image tokens
    actually correctly align with the embeddings and mm feature spec to prevent
    alignment issues when we merge the multimodal embeddings in FMS.
    """
    warmup_inputs = mistral3_mm_utils.get_warmup_inputs(req_count=1)
    warmup_toks = torch.Tensor(warmup_inputs.input_ids)
    warmup_mm_features = warmup_inputs.mm_features[0]

    # Get the total number of expanded image tokens in the inputs
    image_token_id = mistral3_mm_utils.get_multimodal_token_id()
    num_expanded_mm_ids = torch.sum(warmup_toks == image_token_id).item()

    assert num_expanded_mm_ids == warmup_mm_features.mm_position.length


def test_warmup_feature_correctness(mistral3_mm_utils):
    """Ensure that the expanded image token count is actually
    correct with respect to the input image size by reprocessing
    it with transformers and comparing the result.

    NOTE: depending on the implementation, this may be redundant.
    """
    image_token_id = mistral3_mm_utils.get_multimodal_token_id()

    warmup_inputs = mistral3_mm_utils.get_warmup_inputs(req_count=1)
    warmup_toks = torch.Tensor(warmup_inputs.input_ids)
    warmup_mm_features = warmup_inputs.mm_features[0]

    num_expanded_mm_ids = torch.sum(warmup_toks == image_token_id).item()

    processor = AutoProcessor.from_pretrained(MISTRAL3_MODEL.name, revision=MISTRAL3_MODEL.revision)
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

    # Check that the image shapes are also correct
    assert bool(torch.all(preproc_res.image_sizes == image_dims))
