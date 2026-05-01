from collections import defaultdict

import pytest
from llm_cache import get_cached_llm
from spyre_util import ModelInfo
from vllm import LLM, SamplingParams

pytestmark = [pytest.mark.chunked_prefill]


def test_spyre_temperature(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )

    prompt = "The capital of the United Kingdom is"
    params1 = SamplingParams(temperature=0.0, seed=8780, max_tokens=20)
    params2 = SamplingParams(temperature=0.5, seed=8780, max_tokens=20)
    params3 = SamplingParams(temperature=1.0, seed=8780, max_tokens=20)

    outputs = spyre_model.generate([prompt, prompt, prompt], [params1, params2, params3])
    output1, output2, output3 = outputs[0], outputs[1], outputs[2]

    assert output1.outputs[0].text != output2.outputs[0].text
    assert output2.outputs[0].text != output3.outputs[0].text


def test_spyre_max_tokens(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )

    prompt = "Count to twenty"
    params = [
        SamplingParams(temperature=0, seed=8780, max_tokens=5),
        SamplingParams(temperature=0, seed=8780, max_tokens=10),
        SamplingParams(temperature=0, seed=8780, max_tokens=1),
        SamplingParams(temperature=0, seed=8780, max_tokens=6),
        SamplingParams(temperature=0, seed=8780, max_tokens=12),
    ]

    outputs = spyre_model.generate([prompt] * len(params), params)

    for i in range(len(params)):
        assert len(outputs[i].outputs[0].token_ids) == params[i].max_tokens


def test_spyre_stop_sequence(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    stop_str = "7"
    prompt = "1 2 3 4 5 "

    params1 = SamplingParams(stop=[stop_str], max_tokens=10, temperature=0)
    params2 = SamplingParams(max_tokens=10, temperature=0)

    outputs = spyre_model.generate([prompt, prompt], [params1, params2])
    output1, output2 = outputs[0], outputs[1]

    assert stop_str not in output1.outputs[0].text
    assert output1.outputs[0].finish_reason == "stop"
    assert output2.outputs[0].finish_reason != "stop"


def max_repetitions(output):
    histo = defaultdict(int)
    for token in output.outputs[0].token_ids:
        histo[token] += 1

    return max(histo.values())


def test_spyre_presence_penalty(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "REPEAT OVER AND OVER AGAIN THE MINIMUM TIMES POSSIBLE: one one one one one"

    param1 = SamplingParams(presence_penalty=2.0, seed=8780, max_tokens=20)
    param2 = SamplingParams(presence_penalty=-2.0, seed=8780, max_tokens=20)

    outputs = spyre_model.generate([prompt, prompt], [param1, param2])
    with_penalty, no_penalty = outputs[0], outputs[1]

    with_penalty_max = max_repetitions(with_penalty)
    no_penalty_max = max_repetitions(no_penalty)

    assert no_penalty_max > 1
    assert no_penalty_max > with_penalty_max


def test_spyre_frequency_penalty(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )

    prompt = "repeat the word hi ten times:"

    param1 = SamplingParams(frequency_penalty=2.0, seed=8780, max_tokens=20)
    param2 = SamplingParams(frequency_penalty=-2.0, seed=8780, max_tokens=20)

    outputs = spyre_model.generate([prompt, prompt], [param1, param2])
    with_penalty, no_penalty = outputs[0], outputs[1]

    with_penalty_max = max_repetitions(with_penalty)
    no_penalty_max = max_repetitions(no_penalty)
    assert no_penalty_max > 1
    assert no_penalty_max > with_penalty_max


def test_spyre_n_generations(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "The three most popular sports in the world are: "

    params = SamplingParams(n=3, seed=8780, max_tokens=20)

    output = spyre_model.generate(prompt, params)[0]

    assert len(output.outputs) == 3
    assert output.outputs[0].text != output.outputs[1].text
    assert output.outputs[1].text != output.outputs[2].text


def token_diversity(
    spyre_model: LLM, prompt: str, params: list[SamplingParams], n_experiments
) -> list[int]:
    """Runs multiple prompts for every sampling param provided. Returns a list of the number of
    unique tokens generated for each sampling param."""
    num_params = len(params)
    token_sets = []

    expanded_params = []
    for param in params:
        expanded_params.extend([param] * n_experiments)

    outputs = spyre_model.generate(
        [prompt] * n_experiments * num_params, expanded_params, use_tqdm=False
    )

    for i in range(num_params):
        param_outputs = outputs[i * n_experiments : (i + 1) * n_experiments]
        token_set = set()
        for output in param_outputs:
            token_set.update(output.outputs[0].token_ids)
        token_sets.append(token_set)

    return [len(token_set) for token_set in token_sets]


def test_spyre_top_p(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "The first three letters of the alphabet are"
    params1 = SamplingParams(top_p=0.01, temperature=1, max_tokens=10)
    params2 = SamplingParams(temperature=1, max_tokens=10)

    token_div1, token_div2 = token_diversity(spyre_model, prompt, [params1, params2], 3)

    assert token_div1 < token_div2


def test_spyre_top_k(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "The opposite of hot is"
    params1 = SamplingParams(temperature=1, top_k=1, max_tokens=5)
    params2 = SamplingParams(temperature=1, max_tokens=5)

    token_div1, token_div2 = token_diversity(spyre_model, prompt, [params1, params2], 3)
    assert token_div1 < token_div2


def test_spyre_logit_bias(
    model: ModelInfo,
    backend,
    monkeypatch,
    use_llm_cache,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    tokenizer = spyre_model.get_tokenizer()
    banned_word = "train"
    forced_word = "plane"

    banned_ids = tokenizer.encode(banned_word, add_special_tokens=False)
    forced_ids = tokenizer.encode(forced_word, add_special_tokens=False)

    banned_word_id = banned_ids[0]
    forced_word_id = forced_ids[0]

    prompt = "The fastest way to travel between continents is by "
    params1 = SamplingParams(
        temperature=0,
        max_tokens=5,
        logit_bias={
            banned_word_id: -100,
            forced_word_id: 100,
        },
    )
    params2 = SamplingParams(temperature=0, max_tokens=5)

    output = spyre_model.generate([prompt, prompt], [params1, params2])

    assert banned_word not in output[0].outputs[0].text.lower()
    assert forced_word in output[0].outputs[0].text.lower()

    assert output[0].outputs[0].text != output[1].outputs[0].text


def test_spyre_min_tokens(
    model: ModelInfo,
    backend,
    monkeypatch,
    use_llm_cache,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "What is the capital of the USA?"
    tokenizer = spyre_model.get_tokenizer()
    eos_id = tokenizer.eos_token_id

    params1 = SamplingParams(min_tokens=10, logit_bias={eos_id: 1000}, seed=8780, max_tokens=20)
    params2 = SamplingParams(seed=8780, logit_bias={eos_id: 1000}, max_tokens=20)

    output = spyre_model.generate([prompt] * 2, [params1, params2])

    # Logits bias should force eos token appears, then we check if
    # after min tokens reached the logits processor is properly
    # cleared. Therefore token count shall be 10 + 1
    # (min_tokens + eos_token_id)
    assert len(output[0].outputs[0].token_ids) == 11
    assert len(output[1].outputs[0].token_ids) == 1


def test_spyre_ignore_eos(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    tokenizer = spyre_model.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    prompt = "This is the end of the story"

    params1 = SamplingParams(ignore_eos=True, logit_bias={eos_id: 50}, seed=8780, max_tokens=20)
    params2 = SamplingParams(ignore_eos=False, logit_bias={eos_id: 50}, seed=8780, max_tokens=20)

    outputs = spyre_model.generate([prompt, prompt], [params1, params2])
    output1, output2 = outputs[0], outputs[1]

    assert len(output1.outputs[0].token_ids) == 20
    assert len(output2.outputs[0].token_ids) != len(output1.outputs[0].token_ids)

    assert output1.outputs[0].finish_reason == "length"
    assert output2.outputs[0].finish_reason != "length"


def test_spyre_min_p(
    model: ModelInfo,
    backend,
    monkeypatch,
    use_llm_cache,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "The opposite of black is"
    params1 = SamplingParams(min_p=0.5, temperature=1, max_tokens=5)
    params2 = SamplingParams(temperature=1, max_tokens=5)

    token_div1, token_div2 = token_diversity(spyre_model, prompt, [params1, params2], 3)

    assert token_div1 < token_div2


def test_spyre_bad_words(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "The capital of France is"
    params1 = SamplingParams(
        max_tokens=5, temperature=0, bad_words=[" Paris", " Parisi", " France"]
    )
    params2 = SamplingParams(max_tokens=5, temperature=0)

    outputs = spyre_model.generate([prompt, prompt], [params1, params2])
    output1, output2 = outputs[0], outputs[1]

    assert "Paris" not in output1.outputs[0].text
    assert "France" not in output1.outputs[0].text
    assert output1.outputs[0].text != output2.outputs[0].text


def test_spyre_detokenize(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    prompt = "Hello, world!"
    params = SamplingParams(max_tokens=5, temperature=0, detokenize=False)
    output = spyre_model.generate(prompt, params)[0]

    assert output.outputs[0].text == ""
    assert len(output.outputs[0].token_ids) > 0


def test_spyre_logprobs(
    model: ModelInfo,
    backend,
    monkeypatch,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    use_llm_cache,
):
    spyre_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        backend=backend,
        monkeypatch=monkeypatch,
    )
    num_logprobs = 5
    prompt = "The sky is"
    params = SamplingParams(max_tokens=5, temperature=0, logprobs=num_logprobs)
    output = spyre_model.generate(prompt, params)[0]

    completion_output = output.outputs[0]

    assert completion_output.logprobs is not None
    assert len(completion_output.logprobs) == len(completion_output.token_ids)
    assert len(completion_output.logprobs[0]) == num_logprobs
