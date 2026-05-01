"""Tests for ModelMatcher - pattern-based model matching logic."""

import pytest
from unittest.mock import Mock, seal

from sendnn_inference.config.model_config import ArchitecturePattern
from sendnn_inference.config.model_matcher import ModelMatcher

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def matcher():
    """Fixture providing a ModelMatcher instance."""
    return ModelMatcher()


class TestModelMatcherHappyPath:
    """Tests for successful pattern matching scenarios."""

    def test_match_by_model_type_only(self, matcher):
        """Test matching with only model_type specified (minimal pattern)."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="llama",
            attributes={},
        )
        hf_config = Mock(model_type="llama", num_layers=32, hidden_size=4096)

        assert matcher.matches(hf_config, pattern)

    def test_match_model_with_multiple_attributes(self, matcher):
        """Test matching model with multiple attributes (precise pattern)."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
            },
        )
        hf_config = Mock(
            model_type="granite",
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=50000,  # Extra attribute not in pattern
        )

        assert matcher.matches(hf_config, pattern)

    def test_match_model_with_quantization_config_dict(self, matcher):
        """Test matching model with quantization_config dictionary."""
        pattern = ArchitecturePattern(
            model_name="test-model-fp8",
            model_type="granite",
            attributes={
                "quantization_config": {
                    "quant_method": "fp8",
                },
            },
        )
        hf_config = Mock(
            model_type="granite",
            quantization_config={
                "quant_method": "fp8",
                "activation_scheme": "dynamic",  # Extra key not in pattern
            },
        )

        assert matcher.matches(hf_config, pattern)

    @pytest.mark.parametrize("model_type", ["roberta", "xlm-roberta"])
    def test_match_embedding_models(self, matcher, model_type):
        """Test matching embedding models (roberta and xlm-roberta)."""
        pattern = ArchitecturePattern(
            model_name=f"{model_type}-embedding",
            model_type=model_type,
            attributes={"hidden_size": 1024},
        )
        hf_config = Mock(
            model_type=model_type,
            hidden_size=1024,
            num_hidden_layers=24,
        )
        assert matcher.matches(hf_config, pattern)

    def test_match_generation_model_granite(self, matcher):
        """Test matching generation models (granite)."""
        pattern = ArchitecturePattern(
            model_name="granite-8b",
            model_type="granite",
            attributes={
                "num_hidden_layers": 32,
                "hidden_size": 4096,
            },
        )
        hf_config = Mock(
            model_type="granite",
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert matcher.matches(hf_config, pattern)

    def test_match_generation_model_granitemoehybrid(self, matcher):
        """Test matching generation models (granitemoehybrid)."""
        pattern = ArchitecturePattern(
            model_name="granite-moe",
            model_type="granitemoehybrid",
            attributes={
                "num_hidden_layers": 40,
                "num_experts": 16,
            },
        )
        hf_config = Mock(
            model_type="granitemoehybrid",
            num_hidden_layers=40,
            num_experts=16,
            hidden_size=4096,
        )

        assert matcher.matches(hf_config, pattern)


class TestModelMatcherEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_no_match_when_model_type_differs(self, matcher):
        """Test that different model_type causes no match."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="llama",
            attributes={},
        )
        hf_config = Mock(model_type="granite")

        assert not matcher.matches(hf_config, pattern)

    @pytest.mark.parametrize(
        "hf_config_attrs,reason",
        [
            ({"hidden_size": 4096}, "missing"),  # num_hidden_layers missing
            ({"num_hidden_layers": 40}, "different"),  # Different value
        ],
        ids=["attribute_missing", "attribute_value_differs"],
    )
    def test_no_match_when_attribute_mismatch(
        self, matcher, hf_config_attrs, reason, caplog_sendnn_inference
    ):
        """Test that missing or different attribute causes no match."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={"num_hidden_layers": 32},
        )
        # Create Mock and seal to prevent auto-attribute creation
        hf_config = Mock(model_type="granite", **hf_config_attrs)
        seal(hf_config)

        assert not matcher.matches(hf_config, pattern)

        # Verify debug log for missing attribute case
        if reason == "missing":
            assert any(
                "missing attribute" in record.message.lower()
                and "num_hidden_layers" in record.message
                for record in caplog_sendnn_inference.records
            )

    def test_no_match_when_quantization_config_format_differs(self, matcher):
        """Test that non-dict quantization_config causes no match."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={
                "quantization_config": {
                    "quant_method": "fp8",
                },
            },
        )
        hf_config = Mock(
            model_type="granite",
            quantization_config="fp8",  # String instead of dict
        )

        assert not matcher.matches(hf_config, pattern)

    def test_handle_hf_config_without_model_type(self, matcher):
        """Test handling of HF config without model_type attribute."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={},
        )
        hf_config = Mock(spec=[])  # No attributes at all

        assert not matcher.matches(hf_config, pattern)

    @pytest.mark.parametrize(
        "pattern_qconfig,hf_qconfig,should_match",
        [
            # Nested dict matching - extra keys in config are OK
            (
                {"quant_method": "fp8", "activation_scheme": "dynamic"},
                {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_scheme": "per_channel",
                },
                True,
            ),
            # Partial match fails - missing required key
            (
                {"quant_method": "fp8", "activation_scheme": "dynamic"},
                {"quant_method": "fp8"},
                False,
            ),
            # Value mismatch
            (
                {"quant_method": "fp8"},
                {"quant_method": "int8"},
                False,
            ),
        ],
        ids=["nested_dict_with_extra_keys", "missing_required_key", "value_mismatch"],
    )
    def test_quantization_config_matching(self, matcher, pattern_qconfig, hf_qconfig, should_match):
        """Test various quantization config matching scenarios."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={"quantization_config": pattern_qconfig},
        )
        hf_config = Mock(model_type="granite", quantization_config=hf_qconfig)

        assert matcher.matches(hf_config, pattern) == should_match

    def test_empty_attributes_dict_matches_any_config(self, matcher):
        """Test that empty attributes dict matches any config with correct model_type."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={},  # No required attributes
        )
        hf_config = Mock(
            model_type="granite",
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        assert matcher.matches(hf_config, pattern)

    def test_multiple_attributes_one_mismatch_fails(self, matcher):
        """Test that one mismatched attribute among many causes failure."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="granite",
            attributes={
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
            },
        )
        hf_config = Mock(
            model_type="granite",
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=64,  # This one is wrong
        )

        assert not matcher.matches(hf_config, pattern)

    def test_match_text_config_ignores_extra_attributes(self, matcher):
        """Test matching model with text_config sub-object and that extra attributes are ignored."""
        pattern = ArchitecturePattern(
            model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            model_type="pixtral",
            attributes={"text_config": {"num_heads": 2}},
        )
        text_config = Mock(num_heads=2, num_layers=40, hidden_size=4096)  # extra attrs OK
        hf_config = Mock(model_type="pixtral", text_config=text_config)

        assert matcher.matches(hf_config, pattern)

    def test_no_match_when_text_config_attribute_value_differs(self, matcher):
        """Test that mismatched text_config attribute value causes no match."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="pixtral",
            attributes={"text_config": {"num_heads": 2}},
        )
        text_config = Mock(num_heads=8)  # wrong value
        hf_config = Mock(model_type="pixtral", text_config=text_config)

        assert not matcher.matches(hf_config, pattern)

    def test_no_match_when_text_config_attribute_missing(self, matcher):
        """Test that missing attribute in text_config sub-object causes no match."""
        pattern = ArchitecturePattern(
            model_name="test-model",
            model_type="pixtral",
            attributes={"text_config": {"num_heads": 2}},
        )
        text_config = Mock(spec=["num_layers"])  # num_heads not present
        hf_config = Mock(model_type="pixtral", text_config=text_config)

        assert not matcher.matches(hf_config, pattern)


# Made with Bob
