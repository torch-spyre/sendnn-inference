"""Unit tests for platform validation of structured outputs.

Tests the fix in vllm_spyre/platform.py that strips structured_outputs
from SamplingParams during request validation.
"""

import sys
import os
from unittest.mock import MagicMock
import pytest
from types import SimpleNamespace
from vllm import SamplingParams
from vllm.inputs import tokens_input
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm_spyre.platform import SpyrePlatform


pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def mock_spyre_config():
    """Mock SpyrePlatform._config for all tests."""
    original_config = SpyrePlatform._config
    mock_config = MagicMock()
    mock_config.model_config.max_model_len = 512
    SpyrePlatform._config = mock_config
    yield mock_config
    SpyrePlatform._config = original_config


class TestStructuredOutputValidation:
    """Test that platform validation strips structured outputs from requests."""

    def test_strips_structured_outputs(self):
        """Test that validate_request sets structured_outputs to None."""
        params = SamplingParams(
            max_tokens=20, structured_outputs=StructuredOutputsParams(json_object=True)
        )

        assert params.structured_outputs is not None

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        assert params.structured_outputs is None

    def test_logs_warning_when_stripping(self, caplog_vllm_spyre):
        """Test that a warning is logged when stripping structured_outputs."""
        params = SamplingParams(
            max_tokens=20, structured_outputs=StructuredOutputsParams(json_object=True)
        )

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        assert len(caplog_vllm_spyre.records) > 0
        warning_record = caplog_vllm_spyre.records[0]
        assert warning_record.levelname == "WARNING"
        assert "Structured outputs" in warning_record.message
        assert "not supported" in warning_record.message

    @pytest.mark.parametrize(
        "structured_output",
        [
            StructuredOutputsParams(json_object=True),
            StructuredOutputsParams(regex="[0-9]+"),
        ],
    )
    def test_strips_different_structured_output_types(self, structured_output):
        """Test validation with different types of structured outputs."""
        params = SamplingParams(max_tokens=20, structured_outputs=structured_output)

        assert params.structured_outputs is not None

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        assert params.structured_outputs is None

    def test_preserves_other_sampling_params(self):
        """Test that other sampling params are not affected by the fix."""
        params = SamplingParams(
            max_tokens=20,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )

        # Store original values
        original_values = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
        }

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        # Verify other params are unchanged
        assert params.max_tokens == original_values["max_tokens"]
        assert params.temperature == original_values["temperature"]
        assert params.top_p == original_values["top_p"]
        assert params.top_k == original_values["top_k"]
        # But structured_outputs should be None
        assert params.structured_outputs is None

    def test_does_not_affect_pooling_params(self):
        """Test that PoolingParams are not affected (early return in validate_request)."""
        pooling_params = PoolingParams()

        # Should not raise any errors and should return early
        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), pooling_params)

        # PoolingParams don't have structured_outputs, so just verify no exception
        assert True  # If we got here, the early return worked


class TestSendnnConfigurationValidation:
    """Test sendnn configuration validation with model_config parameter."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock ModelConfig for testing."""
        mock_config = MagicMock()
        mock_config.runner_type = "generate"
        return mock_config

    @pytest.fixture
    def mock_embedding_model_config(self):
        """Create a mock ModelConfig for embedding models."""
        mock_config = MagicMock()
        mock_config.runner_type = "pooling"
        return mock_config

    def test_skips_validation_for_non_generate_models(
        self, mock_embedding_model_config, monkeypatch
    ):
        """Test that validation is skipped for non-generative models (e.g., embeddings)."""
        # Set up sendnn backend enabled
        monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        SpyrePlatform._torch_sendnn_configured = False

        # Mock torch_sendnn import
        mock_torch_sendnn = MagicMock()
        monkeypatch.setitem(sys.modules, "torch_sendnn", mock_torch_sendnn)

        # Should not raise and should mark as configured
        SpyrePlatform.maybe_ensure_sendnn_configured(mock_embedding_model_config)

        assert SpyrePlatform._torch_sendnn_configured is True

    def test_skips_validation_when_cache_disabled(self, mock_model_config, monkeypatch):
        """Test that validation is skipped when TORCH_SENDNN_CACHE_ENABLE is 0."""
        # Set up sendnn backend enabled
        monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "0")
        SpyrePlatform._torch_sendnn_configured = False

        # Mock torch_sendnn import
        mock_torch_sendnn = MagicMock()
        monkeypatch.setitem(sys.modules, "torch_sendnn", mock_torch_sendnn)

        # Should not raise and should mark as configured
        SpyrePlatform.maybe_ensure_sendnn_configured(mock_model_config)

        assert SpyrePlatform._torch_sendnn_configured is True

    def test_validates_generate_models_with_cache_enabled(self, mock_model_config, monkeypatch):
        """Test that validation runs for generative models with cache enabled."""
        # Set up sendnn backend enabled with cache
        monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "1")
        monkeypatch.setenv("VLLM_DT_CHUNK_LEN", "512")
        monkeypatch.setenv("VLLM_DT_MAX_CONTEXT_LEN", "4096")
        monkeypatch.setenv("VLLM_DT_MAX_BATCH_SIZE", "32")
        monkeypatch.setenv("VLLM_DT_MAX_BATCH_TKV_LIMIT", "8192")
        SpyrePlatform._torch_sendnn_configured = False

        # Mock torch_sendnn with proper backend state
        # Using a `MagicMock` here would be very hard to do because of the `.getattr(__state)`
        # call during validation. This uses `SimpleNamespaces` instead, which allows us to set an
        # arbitrarily nested config dict, but will fail if access is attempted on any other
        # attributes on `torch_sendnn`.
        # 🌶️ This is super nosy and incredibly coupled to the implementation of `torch_sendnn`, but
        # so is the validation code itself. 🌶️
        mock_torch_sendnn = SimpleNamespace(
            backends=SimpleNamespace(
                sendnn_backend=SimpleNamespace(
                    __state=SimpleNamespace(
                        spyre_graph_cache=SimpleNamespace(
                            deeptools_config={
                                "config": {
                                    "vllm_chunk_length": "512",
                                    "vllm_max_context_length": "4096",
                                    "vllm_max_batch_size": "32",
                                    "vllm_max_batch_tkv_limit": "8192",
                                }
                            }
                        )
                    )
                )
            )
        )
        monkeypatch.setitem(sys.modules, "torch_sendnn", mock_torch_sendnn)

        # Should validate successfully
        SpyrePlatform.maybe_ensure_sendnn_configured(mock_model_config)

        assert SpyrePlatform._torch_sendnn_configured is True

    def test_logs_warning_on_backend_state_read_error(
        self, mock_model_config, monkeypatch, caplog_vllm_spyre
    ):
        """Test that warning is logged when backend state cannot be read."""
        # Set up sendnn backend enabled with cache
        monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "1")
        monkeypatch.setenv("VLLM_DT_CHUNK_LEN", "512")
        SpyrePlatform._torch_sendnn_configured = False

        # Mock torch_sendnn with missing backend state (AttributeError)
        mock_torch_sendnn = MagicMock()
        mock_torch_sendnn.backends.sendnn_backend = MagicMock(spec=[])  # No __state attribute
        monkeypatch.setitem(sys.modules, "torch_sendnn", mock_torch_sendnn)

        # Should log warning and continue
        with pytest.raises(AssertionError):  # Will fail validation but should log warning first
            SpyrePlatform.maybe_ensure_sendnn_configured(mock_model_config)

        # Check that warning was logged with exception details
        warning_records = [r for r in caplog_vllm_spyre.records if r.levelname == "WARNING"]
        assert any(
            "Error reading torch_sendnn backend state for validation" in r.message
            for r in warning_records
        )

    def test_flex_device_set_for_sendnn_compile_only(self, monkeypatch):
        """Test that FLEX_DEVICE is set to COMPILE when backend is sendnn_compile_only."""
        # Set up the backend
        monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn_compile_only")

        # Remove FLEX_DEVICE if it exists to ensure clean test
        monkeypatch.delenv("FLEX_DEVICE", raising=False)

        # Create mock configs
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.runner_type = "generate"
        mock_vllm_config.model_config.is_multimodal_model = False
        mock_vllm_config.parallel_config.world_size = 1
        mock_vllm_config.scheduler_config.max_num_batched_tokens = 64
        mock_vllm_config.model_config.max_model_len = 128
        mock_vllm_config.scheduler_config.max_num_seqs = 2

        # Call check_and_update_config which should set FLEX_DEVICE
        SpyrePlatform.check_and_update_config(
            vllm_config=mock_vllm_config,
        )

        # Verify FLEX_DEVICE was set to COMPILE
        assert os.environ.get("FLEX_DEVICE") == "COMPILE"
