import os

import pytest
from pydantic import ValidationError
from vllm import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

from sendnn_inference.platform import SpyrePlatform
from spyre_util import REFERENCE_MODELS, environ_checkpoint


# Test that the default chunk size is 512 when chunked prefill is enabled,
# and that --max-num-batched-tokens overrides this default.
def test_chunk_size_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the sendnn backend to activate the model configurator
    monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn")
    # Mock the torch_sendnn configuration check
    monkeypatch.setattr(SpyrePlatform, "maybe_ensure_sendnn_configured", lambda: None)

    model = REFERENCE_MODELS["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
    common_args = [
        "--model",
        model.name,
        "--revision",
        model.revision,
        "--max-model-len",
        "64",
        "--max-num-seqs",
        "32",
        "-tp",
        "4",
    ]

    with environ_checkpoint():
        # Test default chunk size is 512
        engine_args = _build_engine_args(common_args)
        assert engine_args.max_num_batched_tokens == 512
        vllm_config = engine_args.create_engine_config()
        assert vllm_config.scheduler_config.max_num_batched_tokens == 512
        assert os.environ.get("VLLM_DT_CHUNK_LEN") == "512"

    with environ_checkpoint():
        # Test that --max-num-batched-tokens overrides the default
        engine_args = _build_engine_args([*common_args, "--max-num-batched-tokens", "128"])
        assert engine_args.max_num_batched_tokens == 128
        vllm_config = engine_args.create_engine_config()
        assert vllm_config.scheduler_config.max_num_batched_tokens == 128
        assert os.environ.get("VLLM_DT_CHUNK_LEN") == "128"

    with environ_checkpoint():
        # Test that an invalid value will trigger an error (42 is not a multiple of the block size)
        engine_args = _build_engine_args([*common_args, "--max-num-batched-tokens", "42"])
        assert engine_args.max_num_batched_tokens == 42
        with pytest.raises(ValidationError):
            engine_args.create_engine_config()


def test_prefix_caching_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    model = REFERENCE_MODELS["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
    common_args = [
        "--model",
        model.name,
        "--revision",
        model.revision,
        "--max-model-len",
        "1024",
    ]

    with environ_checkpoint():
        # Verify that prefix caching is on by default with sendnn-inference
        engine_args = _build_engine_args(
            [
                *common_args,
            ]
        )
        assert engine_args.enable_prefix_caching
        vllm_config = engine_args.create_engine_config()
        assert engine_args.enable_prefix_caching
        assert vllm_config.cache_config.enable_prefix_caching

    with environ_checkpoint():
        # Test that it can be disabled
        engine_args = _build_engine_args([*common_args, "--no-enable-prefix-caching"])
        assert not engine_args.enable_prefix_caching
        vllm_config = engine_args.create_engine_config()
        assert not engine_args.enable_prefix_caching
        assert not vllm_config.cache_config.enable_prefix_caching


def _build_engine_args(cli_args: list[str]) -> EngineArgs:
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(cli_args)

    engine_args = EngineArgs.from_cli_args(args)

    return engine_args


def test_cli_max_num_batched_tokens(monkeypatch):
    # Use the sendnn backend to activate the model configurator
    monkeypatch.setenv("SENDNN_INFERENCE_DYNAMO_BACKEND", "sendnn")
    # Mock the torch_sendnn configuration check
    monkeypatch.setattr(SpyrePlatform, "maybe_ensure_sendnn_configured", lambda: None)

    test_model = os.path.join(os.path.dirname(__file__), "bge_copy")

    common_args = [
        "--model",
        test_model,
    ]

    with environ_checkpoint():
        engine_args = _build_engine_args(common_args)
        # this line throws an exception without the chunked prefill reranker fix
        engine_args.create_engine_config()
