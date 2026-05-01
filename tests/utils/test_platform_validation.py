"""Unit tests for platform validation of structured outputs.

Tests the fix in sendnn_inference/platform.py that strips structured_outputs
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
from sendnn_inference.platform import SpyrePlatform


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
    """Test that platform validation passes structured outputs through unchanged."""

    def test_preserves_structured_outputs(self):
        """Test that validate_request does not strip structured_outputs."""
        structured_outputs = StructuredOutputsParams(json_object=True)
        params = SamplingParams(max_tokens=20, structured_outputs=structured_outputs)

        assert params.structured_outputs is not None

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        assert params.structured_outputs is not None

    def test_no_warning_logged_for_structured_outputs(self, caplog_sendnn_inference):
        """Test that no warning is logged when structured_outputs are present."""
        params = SamplingParams(
            max_tokens=20, structured_outputs=StructuredOutputsParams(json_object=True)
        )

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        warning_records = [r for r in caplog_sendnn_inference.records if r.levelname == "WARNING"]
        assert not any(
            "Structured outputs" in r.message and "not supported" in r.message
            for r in warning_records
        )

    @pytest.mark.parametrize(
        "structured_output",
        [
            StructuredOutputsParams(json_object=True),
            StructuredOutputsParams(regex="[0-9]+"),
        ],
    )
    def test_preserves_different_structured_output_types(self, structured_output):
        """Test validation preserves different types of structured outputs."""
        params = SamplingParams(max_tokens=20, structured_outputs=structured_output)

        assert params.structured_outputs is not None

        SpyrePlatform.validate_request(tokens_input(prompt_token_ids=[0]), params)

        assert params.structured_outputs is not None

    def test_preserves_other_sampling_params(self):
        """Test that other sampling params are not affected by validation."""
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

        # Verify all params are unchanged
        assert params.max_tokens == original_values["max_tokens"]
        assert params.temperature == original_values["temperature"]
        assert params.top_p == original_values["top_p"]
        assert params.top_k == original_values["top_k"]
        assert params.structured_outputs is not None

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
        monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn")
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
        monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn")
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
        monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn")
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
        self, mock_model_config, monkeypatch, caplog_sendnn_inference
    ):
        """Test that warning is logged when backend state cannot be read."""
        # Set up sendnn backend enabled with cache
        monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn")
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
        warning_records = [r for r in caplog_sendnn_inference.records if r.levelname == "WARNING"]
        assert any(
            "Error reading torch_sendnn backend state for validation" in r.message
            for r in warning_records
        )

    def test_flex_device_set_for_sendnn_compile_only(self, monkeypatch):
        """Test that FLEX_DEVICE is set to COMPILE when backend is sendnn_compile_only."""
        # Set up the backend
        monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn_compile_only")

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


class TestPreRegisterAndUpdate:
    """Test SpyrePlatform.pre_register_and_update conditional defaults."""

    @pytest.fixture(autouse=True)
    def clear_conditional_defaults(self):
        """Clear conditional defaults before each test to ensure isolation."""
        from sendnn_inference.argparse_utils import ConditionalDefaultManager

        ConditionalDefaultManager.clear()
        yield
        ConditionalDefaultManager.clear()

        # Re-import huggingface_hub constants to reset any patched env vars
        import importlib
        import huggingface_hub.constants

        importlib.reload(huggingface_hub.constants)

    @pytest.fixture
    def arg_parsers(self):
        """Create main parser and serve subparser like vLLM's CLI structure.

        Returns a tuple of (main_parser, serve_subparser). The subparser is
        passed to pre_register_and_update, but parse_args should be called
        on the main parser.
        """
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        # Create main parser like vLLM's CLI
        main_parser = FlexibleArgumentParser()
        subparsers = main_parser.add_subparsers(dest="subcommand")

        # Create serve subparser with the actual arguments
        serve_parser = subparsers.add_parser("serve", help="Serve model")
        serve_parser.add_argument("--config-format", dest="config_format", default="auto")
        serve_parser.add_argument("--tokenizer-mode", dest="tokenizer_mode", default="auto")
        serve_parser.add_argument("--revision", dest="revision", default=None)
        serve_parser.add_argument("--hf-token", dest="hf_token", default=None)
        serve_parser.add_argument("model_tag", nargs="?", default=None)

        return main_parser, serve_parser

    def test_config_format_defaults_to_mistral_when_params_json_exists(self, tmp_path, arg_parsers):
        """Test that config_format defaults to 'mistral' when model_tag points to dir with
        params.json."""
        main_parser, serve_parser = arg_parsers

        # Create a temporary directory with params.json (NB: path does not have mistral in the name)
        model_dir = tmp_path / "some_model"
        model_dir.mkdir()
        params_file = model_dir / "params.json"
        params_file.write_text('{"dim": 4096, "n_layers": 32}')

        # Pass only the subparser to pre_register_and_update (like vLLM does)
        SpyrePlatform.pre_register_and_update(serve_parser)
        # But call parse_args on the main parser (like vLLM does)
        args = main_parser.parse_args(["serve", str(model_dir)])

        assert args.config_format == "mistral"
        assert args.tokenizer_mode == "mistral"

    def test_config_format_defaults_to_auto_for_models_without_params_json(
        self, tmp_path, arg_parsers
    ):
        """Test that config_format defaults to 'auto' for models without params.json."""
        main_parser, serve_parser = arg_parsers

        # Create a temporary directory without params.json
        model_dir = tmp_path / "some_other_model"
        model_dir.mkdir()

        # Put some other files in it
        (model_dir / "config.json").write_text('{"_name_or_path": "gpt2"}')
        (model_dir / "pytorch_model.bin").touch()

        # Pass only the subparser to pre_register_and_update (like vLLM does)
        SpyrePlatform.pre_register_and_update(serve_parser)
        # But call parse_args on the main parser (like vLLM does)
        args = main_parser.parse_args(["serve", str(model_dir)])

        assert args.config_format == "auto"
        assert args.tokenizer_mode == "auto"

    def test_explicit_config_format_not_overridden(self, tmp_path, arg_parsers):
        """Test that user-provided config_format is not overridden."""
        main_parser, serve_parser = arg_parsers

        # Create a temporary directory with params.json
        model_dir = tmp_path / "mistral_model"
        model_dir.mkdir()
        params_file = model_dir / "params.json"
        params_file.write_text('{"dim": 4096, "n_layers": 32}')

        # Pass only the subparser to pre_register_and_update (like vLLM does)
        SpyrePlatform.pre_register_and_update(serve_parser)
        # But call parse_args on the main parser (like vLLM does)
        args = main_parser.parse_args(["serve", "--config-format", "hf", str(model_dir)])

        # User explicitly set config_format to "hf", it should not be overridden
        assert args.config_format == "hf"
        # tokenizer_mode should still be set (since it depends on the same logic)
        assert args.tokenizer_mode == "mistral"

    def test_config_format_from_cached_hf_model_offline_mode(
        self, tmp_path, arg_parsers, monkeypatch
    ):
        """Test that config_format is detected from cached HF model in offline mode.

        This creates a mock HF hub cache structure with a mistral model that has
        params.json cached, then runs with HF_HUB_OFFLINE=1 to verify the
        detection works correctly.
        """
        main_parser, serve_parser = arg_parsers

        # Create a mock HF hub cache structure
        # Format: <cache_dir>/models--<org>--<model>/snapshots/<commit_hash>/<files>
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
        repo_folder = cache_dir / "models--mistralai--Mistral-7B-Instruct-v0.1"
        refs_folder = repo_folder / "refs"
        snapshots_folder = repo_folder / "snapshots"

        refs_folder.mkdir(parents=True)
        snapshots_folder.mkdir(parents=True)

        # Create a fake commit hash "main" reference
        (refs_folder / "main").write_text("abc123def456789")

        # Create snapshot directory with the fake commit hash
        snapshot_folder = snapshots_folder / "abc123def456789"
        snapshot_folder.mkdir()

        # Create params.json in the snapshot (this marks it as a mistral model)
        (snapshot_folder / "params.json").write_text('{\n  "dim": 4096,\n  "n_layers": 32\n}')
        # Also create other typical model files
        (snapshot_folder / "config.json").write_text('{"model_type": "mistral"}')

        # Set up the cache environment and offline mode
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        # Re-import huggingface_hub constants to pick up the new env vars
        import importlib
        import huggingface_hub.constants

        importlib.reload(huggingface_hub.constants)

        # Pass only the subparser to pre_register_and_update
        SpyrePlatform.pre_register_and_update(serve_parser)
        # Call parse_args on the main parser with the repo_id
        args = main_parser.parse_args(["serve", repo_id])

        assert args.config_format == "mistral"
        assert args.tokenizer_mode == "mistral"
