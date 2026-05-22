import openai
import pytest

from spyre_util import ModelInfo, RemoteOpenAIServer, get_chicken_soup_prompts


def _check_result(client, model, max_tokens=8, temperature=0.0, n=1) -> None:
    completion = client.completions.create(
        model=model,
        prompt="Hello World!",
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    assert len(completion.choices) == n
    assert len(completion.choices[0].text) > 0


def test_openai_serving(
    remote_openai_server,
    model,
    backend,
    tp_size,
    mode,
    max_num_seqs,
    max_model_len,
):
    """Test online serving using the `vllm serve` CLI"""

    client = remote_openai_server.get_client()
    model = model.name

    _check_result(client, model, n=1)
    _check_result(client, model, temperature=1.0, n=2)

    # Check some basic error handling as well. This is all done in one test
    # now to avoid server boot-up overhead to test each case.

    # FIXME: Add matching strings here for the user-facing error message
    with pytest.raises(openai.APIError):
        # Prompt too long should raise
        long_prompt = "Hello " * max_model_len * 2
        client.completions.create(model=model, prompt=long_prompt, max_tokens=1)

    # Short prompt under context length but requesting too many tokens should raise
    with pytest.raises(openai.APIError):
        client.completions.create(model=model, prompt="Hello World!", max_tokens=max_model_len * 2)


@pytest.mark.parametrize("backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_api_generates_correct_max_tokens(
    remote_openai_server: RemoteOpenAIServer,
    model: ModelInfo,
    backend: str,
    max_model_len: int,
    max_num_seqs: int,
    mode: bool,
):
    """Verify API generates the correct numbers of tokens with CB enabled"""

    client = remote_openai_server.get_client()
    max_tokens = 10

    response = client.completions.create(
        model=model.name, prompt=get_chicken_soup_prompts(1), max_tokens=max_tokens, temperature=0
    )

    assert response.usage.completion_tokens == max_tokens
