import json
import random

import pytest
import torch
from spyre_util import ModelInfo
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdateBuilder, LogitsProcessor

from sendnn_inference.v1.sample.golden_token_injector import GoldenTokenInjector


class DummyVllmConfig:
    def __init__(self, model_config: ModelInfo):
        self.model_config = DummyModelConfig(model_config.name, model_config.revision)


class DummyModelConfig:
    def __init__(self, model: str, revision: str | None):
        self.model = model
        self.tokenizer = model
        self.revision = revision
        self.tokenizer_revision = revision
        self.tokenizer_mode = "hf"
        self.trust_remote_code = True


def step(
    batch_update_builder: BatchUpdateBuilder,
    lp: LogitsProcessor,
    logits: torch.Tensor,
    batch_output_tokens: list[list[int]],
):
    assert logits.shape[0] == len(batch_output_tokens)

    # This is called at each execute model in spyre model runner update_states
    lp.update_state(None)

    out_logits = lp.apply(logits.clone())

    for idx, output_tokens in enumerate(batch_output_tokens):
        # just append a random token
        token_idx = out_logits[idx].argmax(dim=-1).reshape(-1).item()
        output_tokens.append(token_idx)

    return out_logits


def generate_logits(vocab_size: int, batch_size: int = 1):
    return torch.tensor(list(range(vocab_size)) * batch_size, dtype=torch.float32).reshape(
        (batch_size, vocab_size)
    )


@pytest.mark.cpu
@pytest.mark.parametrize("arg_as_string", [True, False])
def test_git_basic_correctness(model: ModelInfo, arg_as_string: bool):
    device = torch.device("cpu")
    git = GoldenTokenInjector(DummyVllmConfig(model), device, False)

    vocab_size = git.tokenizer.vocab_size

    batch_update_builder = BatchUpdateBuilder()
    batch_output_tokens = []
    logits = generate_logits(vocab_size, 1)

    expected_tokens_count = 8

    expected_token_ids = [random.randint(0, vocab_size) for _ in range(expected_tokens_count)]
    git_args = {
        "expected_token_ids": expected_token_ids,
    }

    if arg_as_string:
        git_args = json.dumps(git_args)

    params = SamplingParams(extra_args={"golden_token_injector": git_args})

    prompt_tokens = [random.randint(0, vocab_size) for _ in range(8)]
    batch_update_builder.added.append((0, params, prompt_tokens, batch_output_tokens))
    batch_update = batch_update_builder.get_and_reset(1)
    git.update_state(batch_update)

    for current_idx in range(expected_tokens_count):
        logits = generate_logits(vocab_size, 1)
        step(batch_update_builder, git, logits, [batch_output_tokens])
        assert batch_output_tokens[current_idx] == expected_token_ids[current_idx]


@pytest.mark.cpu
def test_git_out_of_range_expected_tokens(model: ModelInfo):
    # TODO: this test is a huge copy paste from the above
    # improve it later to better reuse
    device = torch.device("cpu")
    git = GoldenTokenInjector(DummyVllmConfig(model), device, False)

    vocab_size = git.tokenizer.vocab_size

    batch_update_builder = BatchUpdateBuilder()
    batch_output_tokens = []
    logits = generate_logits(vocab_size, 1)

    expected_tokens_count = 8

    expected_token_ids = [random.randint(0, vocab_size) for _ in range(expected_tokens_count)]
    git_args = {
        "expected_token_ids": expected_token_ids,
    }

    params = SamplingParams(extra_args={"golden_token_injector": git_args})

    prompt_tokens = [random.randint(0, vocab_size) for _ in range(8)]
    batch_update_builder.added.append((0, params, prompt_tokens, batch_output_tokens))
    batch_update = batch_update_builder.get_and_reset(1)
    git.update_state(batch_update)

    # Inject correctly
    for current_idx in range(expected_tokens_count):
        logits = generate_logits(vocab_size, 1)
        step(batch_update_builder, git, logits, [batch_output_tokens])
        assert batch_output_tokens[current_idx] == expected_token_ids[current_idx]

    # Cannot inject anymore
    logits = generate_logits(vocab_size, 1)
    out_logits = step(batch_update_builder, git, logits, [batch_output_tokens])
    # Keep logits same
    assert torch.allclose(logits, out_logits)
