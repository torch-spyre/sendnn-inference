import os

import pytest
import requests
from spyre_util import ModelInfo, RemoteOpenAIServer, patch_environment
from vllm import LLM, SamplingParams


def _model_info_from_env() -> ModelInfo:
    model = os.environ["SENDNN_INFERENCE_SMOKE_MODEL"]
    revision = os.environ["SENDNN_INFERENCE_SMOKE_MODEL_REV"]
    return ModelInfo(name=model, revision=revision)


@pytest.mark.smoke
def test_decoder_model_load_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model_info_from_env()
    patch_environment("eager", monkeypatch)

    llm = LLM(
        model=model.name,
        revision=model.revision,
        tokenizer=model.name,
        tokenizer_revision=model.revision,
        max_model_len=256,
        max_num_seqs=1,
        enforce_eager=True,
    )

    outputs = llm.generate(
        ["Say hi in two words."],
        SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
    )

    assert len(outputs) == 1
    assert outputs[0].outputs
    assert outputs[0].outputs[0].text.strip()


@pytest.mark.smoke
def test_embedding_model_load_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model_info_from_env()
    patch_environment("eager", monkeypatch, warmup_shapes=[(64, 4)])

    llm = LLM(
        model=model.name,
        revision=model.revision,
        tokenizer=model.name,
        tokenizer_revision=model.revision,
        max_model_len=256,
        enforce_eager=True,
    )

    outputs = llm.embed(["The quick brown fox jumps over the lazy dog."])

    assert len(outputs) == 1
    assert outputs[0].outputs.embedding
    assert len(outputs[0].outputs.embedding) > 0


@pytest.mark.smoke
def test_scoring_model_load_smoke() -> None:
    model = _model_info_from_env()

    with RemoteOpenAIServer(
        model,
        [],
        env_dict={
            "SENDNN_INFERENCE_DYNAMO_BACKEND": "eager",
            "SENDNN_INFERENCE_WARMUP_PROMPT_LENS": "64",
            "SENDNN_INFERENCE_WARMUP_BATCH_SIZES": "4",
        },
        max_wait_seconds=300,
    ) as server:
        response = requests.post(
            server.url_for("/score"),
            json={
                "text_1": "What is the capital of France?",
                "text_2": ["Paris is the capital of France."],
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()

    assert payload["data"]
    assert len(payload["data"]) == 1
    assert isinstance(payload["data"][0]["score"], float)


# Made with Bob
