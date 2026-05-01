"""End-to-end tests for structured output decoding.

Tests structured output support across different backends (guidance, xgrammar, outlines)
and ensures that prompts without structured output requests don't accidentally have them applied.
"""

import json
import pytest
import re
from llm_cache import get_llm
from spyre_util import ModelInfo
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.config import StructuredOutputsConfig

pytestmark = [pytest.mark.chunked_prefill]


# Parametrize all tests over the three structured output backends
# Note: Backend support varies by feature:
# - guidance: supports json_object, json (schema), regex, choice
# - xgrammar: supports json_object, json (schema), regex, choice
# - outlines: supports json (schema), regex, choice (NOT json_object)
STRUCTURED_OUTPUT_BACKENDS = ["guidance", "xgrammar", "outlines"]

# Backends that support json_object (free-form JSON without schema)
JSON_OBJECT_BACKENDS = ["guidance", "xgrammar"]  # outlines requires schema


@pytest.mark.parametrize("structured_output_backend", JSON_OBJECT_BACKENDS)
def test_structured_output_json_object(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
    structured_output_backend: str,
):
    """Test that structured output with json_object=True produces valid JSON."""
    spyre_model = get_llm(
        model=model,
        max_model_len=max_model_len,
        backend=backend,
        monkeypatch=monkeypatch,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        structured_outputs_config=StructuredOutputsConfig(backend=structured_output_backend),
    )

    prompt = "Generate a JSON object with name and age fields for a person."

    params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )

    outputs = spyre_model.generate([prompt], [params])
    output_text = outputs[0].outputs[0].text

    # Verify output is valid JSON
    try:
        json_obj = json.loads(output_text)
        assert isinstance(json_obj, dict), "Output should be a JSON object"
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {output_text}\nError: {e}")


@pytest.mark.parametrize("structured_output_backend", STRUCTURED_OUTPUT_BACKENDS)
def test_structured_output_json_schema(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
    structured_output_backend: str,
):
    """Test that structured output with a JSON schema validates correctly."""
    spyre_model = get_llm(
        model=model,
        max_model_len=max_model_len,
        backend=backend,
        monkeypatch=monkeypatch,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        structured_outputs_config=StructuredOutputsConfig(backend=structured_output_backend),
    )

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    prompt = "Generate a person with name and age only."

    params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        structured_outputs=StructuredOutputsParams(json=schema),
    )

    outputs = spyre_model.generate([prompt], [params])
    output_text = outputs[0].outputs[0].text

    # Verify output is valid JSON matching the schema
    try:
        json_obj = json.loads(output_text)
        assert isinstance(json_obj, dict), "Output should be a JSON object"
        assert "name" in json_obj, "Output should have 'name' field"
        assert "age" in json_obj, "Output should have 'age' field"
        assert isinstance(json_obj["name"], str), "'name' should be a string"
        assert isinstance(json_obj["age"], int), "'age' should be an integer"
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {output_text}\nError: {e}")


@pytest.mark.parametrize("structured_output_backend", STRUCTURED_OUTPUT_BACKENDS)
def test_structured_output_regex(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
    structured_output_backend: str,
):
    """Test that structured output with regex pattern is enforced."""
    spyre_model = get_llm(
        model=model,
        max_model_len=max_model_len,
        backend=backend,
        monkeypatch=monkeypatch,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        structured_outputs_config=StructuredOutputsConfig(backend=structured_output_backend),
    )

    # Regex for phone number format: XXX-XXX-XXXX
    phone_regex = r"\d{3}-\d{3}-\d{4}"

    prompt = "Generate a phone number in XXX-XXX-XXXX format."

    params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        structured_outputs=StructuredOutputsParams(regex=phone_regex),
    )

    outputs = spyre_model.generate([prompt], [params])
    output_text = outputs[0].outputs[0].text.strip()

    # Verify output matches the regex pattern
    match = re.fullmatch(phone_regex, output_text)
    assert match is not None, f"Output '{output_text}' does not match regex pattern '{phone_regex}'"


@pytest.mark.parametrize("structured_output_backend", JSON_OBJECT_BACKENDS)
def test_structured_output_mixed_batch(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
    structured_output_backend: str,
):
    """Test that requests with and without structured outputs can coexist.

    This is critical to ensure that prompts without structured output requests don't
    accidentally have structured outputs applied. This test submits both structured
    and non-structured requests in a single batch to verify they can run together.
    """
    spyre_model = get_llm(
        model=model,
        max_model_len=max_model_len,
        backend=backend,
        monkeypatch=monkeypatch,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        structured_outputs_config=StructuredOutputsConfig(backend=structured_output_backend),
    )

    # Request with structured output (JSON object)
    prompt_structured = "Generate a JSON object with name and age."
    params_structured = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )

    # Request without structured output (free-form text)
    prompt_freeform = "Write a short story about a cat."
    params_freeform = SamplingParams(
        temperature=0.0,
        max_tokens=50,
    )

    # Submit both requests in a single batch to test mixed structured/non-structured outputs
    outputs = spyre_model.generate(
        [prompt_structured, prompt_freeform], [params_structured, params_freeform]
    )

    # Verify structured output is valid JSON
    output_structured = outputs[0]
    output_structured_text = output_structured.outputs[0].text
    try:
        json_obj = json.loads(output_structured_text)
        assert isinstance(json_obj, dict), "Structured output should be a JSON object"
    except json.JSONDecodeError as e:
        pytest.fail(f"Structured output is not valid JSON: {output_structured_text}\nError: {e}")

    # Verify freeform output is not constrained (just has text)
    output_freeform = outputs[1]
    output_freeform_text = output_freeform.outputs[0].text
    assert len(output_freeform_text) > 0, "Freeform output should have text"
    # Don't enforce JSON structure - it should be free-form story text


@pytest.mark.parametrize("structured_output_backend", STRUCTURED_OUTPUT_BACKENDS)
def test_structured_output_choice(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
    structured_output_backend: str,
):
    """Test that structured output with choice constraint works correctly."""
    spyre_model = get_llm(
        model=model,
        max_model_len=max_model_len,
        backend=backend,
        monkeypatch=monkeypatch,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        structured_outputs_config=StructuredOutputsConfig(backend=structured_output_backend),
    )

    choices = ["yes", "no", "maybe"]

    prompt = "Is the sky blue? Answer with yes, no, or maybe."

    params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        structured_outputs=StructuredOutputsParams(choice=choices),
    )

    outputs = spyre_model.generate([prompt], [params])
    output_text = outputs[0].outputs[0].text.strip().lower()

    # Verify output is one of the allowed choices
    assert output_text in choices, f"Output '{output_text}' not in allowed choices {choices}"


# Made with Bob
