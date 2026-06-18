"""Verification of the correctness of the step-by-step execution of chunked
prefill. It does so by comparing, at every engine step (i.e. prefill or decode
iteration), a bunch of attributes. This allows a finer testing of the padding
and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_cp_inference_steps.py`.
"""

import pytest
from scheduling_utils import (
    check_scheduler_inference_steps,
    create_request_for_scheduler_test,
    random_prompt,
    validate_scheduler_steps,
)
from spyre_util import ModelInfo


@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])  # restricted to violate scheduler condition
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefill_tkv_too_big(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Here we ensure that the tkv never goes beyond max_model_len, even in an
    edge case.

    Edge case: due to a long-prompt joining the decode batch, the currently
    decoding request needs to be left-padded, bringing the max-tokens beyond
    max-model-len. We make sure the left-padding is removed on time when
    expanding to a new block, keeping the tkv in acceptable range always.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 60, max tokens = 10, step joining = 0
            * 1: len = 111, max tokens = 17, step joining = 0
    """

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=10,
        prompt=random_prompt(model=model, seed=0, length=60),
        use_golden_token_injection=True,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=17,
        prompt=random_prompt(model=model, seed=0, length=111),
        use_golden_token_injection=True,
    )

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            "step": 1,
            "tkv": 60,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequence 0
            "step": 2,
            "tkv": 61,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 1,
        },
        {
            # Prefill sequence 1
            # Due to left-padding of sequence 0, we now have tkv = 64 + 61
            "step": 3,
            "tkv": 125,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 3,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequences 0 and 1
            "step": 4,
            "tkv": 126,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 3,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequences 0 and 1
            # Last step before tkv would overflow max_context_length
            "step": 6,
            "tkv": 128,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 3,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequences 0 and 1
            # Sequence 0 now needs two blocks. Instead of adding one on the
            # right (which would overflow the tkv), we remove it's left-padding
            # block, bringing back the tkv to a satisfactory value
            "step": 7,
            "tkv": 115,  # corresponds now to tkv of request 1
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 4,
            "n_reserved_blocks": 0,
        },
        {
            # Decode sequences 0 and 1
            # Sequence 0 finishes
            "step": 11,
            "tkv": 119,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_used_blocks": 2,
            "n_reserved_blocks": 0,
        },
        {
            # Decode sequences 1
            "step": 12,
            "tkv": 120,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_used_blocks": 2,
            "n_reserved_blocks": 0,
        },
        {
            # Sequence 1 finishes
            "step": 19,
            "tkv": 127,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 20,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_requests_exceed_batch_tkv_limit(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Scenario where a request cannot be scheduled right away as the
    max batch x tkv limit, e.g the volumetric limit, is exceeded.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 64, max tokens = 2, step joining = 0
            * 2: len = 65, max tokens = 2, step joining = 0
    """

    seqs_max_tokens = [2, 2]
    prompts_lengths = [64, 65]
    steps_add_reqs = [0, 0]
    # total number of blocks needed if scheduled together: (1 + 1)+(1 + 1) = 4
    # note that as not scheduled together, we only needs 2 blocks here
    # needs 2 * (64 + 1) = 2 * 65 = 130
    max_batch_tkv_limit = 129  # not big enough

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 1,
        },
        # Note: we cannot prefill seq 1 as the volumetric limit
        # max_batch_tkv_limit is exceeded: 129 < 130
        # -> cond5 in can_schedule() is False
        {
            # Decode sequence 0
            # Sequence 0 finishes at step 2
            # total blocks in use: 2
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2
            "step": 3,
            "tkv": 65,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_used_blocks": 2,  # 2 - 2 + 2
            "n_reserved_blocks": 0,
        },
        {
            # Decode sequence 1
            # Sequence 1 finishes at step 4
            # total blocks in use: 3
            "step": 4,
            "tkv": 66,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            # total blocks in use: 2 - 2 = 0
            "step": 5,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_batch_tkv_limit=max_batch_tkv_limit,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [514, 1024])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_single_cp_prefill(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Scenario to test the most basic execution of chunked scheduling:
    a single prompts larger than the chunk size.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 1
            * 0: len = 512, max tokens = 1, step joining = 0
    """
    # max_model_len=514 tests an edge case in the scheduler, but does not work
    # on sendnn
    if backend == "sendnn" and max_model_len == 514:
        pytest.skip("sendnn backend with 514 context length will not work")
    if backend != "sendnn" and max_model_len == 1024:
        pytest.skip("skipping 1024 context length test case for CPU to save test time")

    seqs_max_tokens = [2]
    prompts_lengths = [512]
    steps_add_reqs = [0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0 chunk 0
            "step": 1,
            "tkv": 512,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_used_blocks": 2,
            "n_reserved_blocks": 7,
        },
        {
            # Prefill sequence 0 chunk 1
            "step": 2,
            "tkv": 512,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_used_blocks": 4,
            "n_reserved_blocks": 5,
        },
        {
            # Prefill sequence 0 chunk 2
            # total blocks in use: 6
            "step": 3,
            "tkv": 512,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_used_blocks": 6,
            "n_reserved_blocks": 3,
        },
        {
            # Prefill sequence 0 chunk 3
            # total blocks in use: 8
            "step": 4,
            "tkv": 512,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 8,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequence 0
            # seq 0 finishes in this step
            "step": 5,
            "tkv": 513,
            "waiting": [],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 6,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_cp_prefill_interleave1(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Scenario where two sequences are scheduled from the beginning
    and the shorter sequence gets scheduled first. After a couple of
    iterations the interleaving of requests starts.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 1
            * 0: len = 10,  max tokens = 8, step joining = 0
            * 1: len = 512, max tokens = 4, step joining = 0
    """

    seqs_max_tokens = [8, 4]
    prompts_lengths = [10, 512]
    steps_add_reqs = [0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Request 0 is prefilled with a single chunk
            # Token 0 is generated
            "step": 1,
            "tkv": 10,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Request 0 starts decoding.
            # Request 1 can't be scheduled do to a restriction on
            # consecutive prefills. Token 1 is generated
            "step": 2,
            "tkv": 11,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Chunk 0 of request 1 prefill
            "step": 3,
            "tkv": 11,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 3,
            "n_reserved_blocks": 7,
        },
        {
            # Decode 2 of request 0.
            "step": 4,
            "tkv": 12,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 3,
            "n_reserved_blocks": 7,
        },
        {
            # Chunk 1 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 5,
            "tkv": 12,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 5,
            "n_reserved_blocks": 5,
        },
        {
            # Decode 3 of request 0.
            "step": 6,
            "tkv": 13,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 5,
            "n_reserved_blocks": 5,
        },
        {
            # Chunk 2 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 7,
            "tkv": 13,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 7,
            "n_reserved_blocks": 3,
        },
        {
            # Decode 4 of request 0.
            "step": 8,
            "tkv": 14,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 7,
            "n_reserved_blocks": 3,
        },
        {
            # Chunk 3 of request 1 prefill.
            # First token is generated
            # tkv updated for last chunk
            "step": 9,
            "tkv": 512,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 9,
            "n_reserved_blocks": 1,
        },
        {
            # Decode 5 of request 0.
            # Decode 1 of request 1.
            "step": 10,
            "tkv": 527,  # tkv jump
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 6 of request 0.
            # Decode 2 of request 1.
            "step": 11,
            "tkv": 528,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 7 of request 0.
            # Decode 3 of request 1.
            # Requests are done
            "step": 12,
            "tkv": 529,
            "waiting": [],
            "running": [],
            "finished_requests": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 13,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_cp_prefill_no_interleave(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Same as test_cp_prefill_interleave1 but with interleaving disabled

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 1
            * 0: len = 10,  max tokens = 8, step joining = 0
            * 1: len = 512, max tokens = 4, step joining = 0
    """

    monkeypatch.setenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "0")

    seqs_max_tokens = [8, 4]
    prompts_lengths = [10, 512]
    steps_add_reqs = [0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Request 0 is prefilled with a single chunk
            # Token 0 is generated
            "step": 1,
            "tkv": 10,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Chunk 0 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 2,
            "tkv": 10,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 3,
            "n_reserved_blocks": 7,
        },
        {
            # Chunk 1 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 3,
            "tkv": 10,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 5,
            "n_reserved_blocks": 5,
        },
        {
            # Chunk 2 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 4,
            "tkv": 10,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 7,
            "n_reserved_blocks": 3,
        },
        {
            # Chunk 3 of request 1 prefill.
            # First token is generated
            # tkv updated for last chunk
            "step": 5,
            "tkv": 512,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 9,
            "n_reserved_blocks": 1,
        },
        {
            # Both requests start decoding.
            "step": 6,
            "tkv": 523,  # tkv jump
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 2
            "step": 7,
            "tkv": 524,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 3
            "step": 8,
            "tkv": 525,
            "waiting": [],
            "running": ["0"],
            "finished_requests": ["1"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 4 of request 0.
            "step": 9,
            "tkv": 14,  # tkv jump
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 5 of request 0.
            "step": 10,
            "tkv": 15,  # tkv jump
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 6 of request 0.
            "step": 11,
            "tkv": 16,  # tkv jump
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 7 of request 0.
            "step": 12,
            "tkv": 17,  # tkv jump
            "waiting": [],
            "running": [],
            "finished_requests": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 13,  # with or without interleaving we finish at step 13
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_cp_prefill_interleave2(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Same as test_cp_prefill_interleave1 but now the second
    request arrives during decode of step 0

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 1
            * 0: len = 10,  max tokens = 8, step joining = 0
            * 1: len = 512, max tokens = 4, step joining = 3
    """

    seqs_max_tokens = [8, 4]
    prompts_lengths = [10, 512]
    steps_add_reqs = [0, 3]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Request 0 is prefilled with a single chunk
            # Token 0 is generated
            "step": 1,
            "tkv": 10,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 1 of request 0.
            "step": 2,
            "tkv": 11,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 2 of request 0.
            "step": 3,
            "tkv": 12,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 0,
        },
        {
            # Chunk 0 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 4,
            "tkv": 12,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 3,
            "n_reserved_blocks": 7,
        },
        {
            # Decode 3 of request 0.
            "step": 5,
            "tkv": 13,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 3,
            "n_reserved_blocks": 7,
        },
        {
            # Chunk 1 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 6,
            "tkv": 13,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 5,
            "n_reserved_blocks": 5,
        },
        {
            # Decode 4 of request 0.
            "step": 7,
            "tkv": 14,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 5,
            "n_reserved_blocks": 5,
        },
        {
            # Chunk 2 of request 1 prefill
            # tkv of decode batch (tkv not updated until last chunk)
            "step": 8,
            "tkv": 14,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 7,
            "n_reserved_blocks": 3,
        },
        {
            # Decode 5 of request 0.
            "step": 9,
            "tkv": 15,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 7,
            "n_reserved_blocks": 3,
        },
        {
            # Chunk 3 of request 1 prefill.
            # First token is generated
            # tkv updated for last chunk
            "step": 10,
            "tkv": 512,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 9,
            "n_reserved_blocks": 1,
        },
        {
            # Decode 6 of request 0.
            # Decode 1 of request 1.
            "step": 11,
            "tkv": 528,  # tkv jump
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
            "n_reserved_blocks": 0,
        },
        {
            # Decode 7 of request 0.
            # Decode 2 of request 1.
            "step": 12,
            "tkv": 529,
            "waiting": [],
            "running": ["1"],
            "finished_requests": ["0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 9,
            "n_reserved_blocks": 0,
        },
        {
            "step": 13,
            "tkv": 515,  # tkv jump
            "waiting": [],
            "running": [],
            "finished_requests": ["1"],
            "request_outputs": ["1"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 14,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [64])
@pytest.mark.parametrize("available_blocks", [2])
def test_requests_not_enough_blocks(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Scenario where the number of blocks is smaller than the maximum batch
    size times the maximum number of blocks per sequence. This means that we
    cannot schedule all the requests at once

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 64, max tokens = 2, step joining = 0
            * 2: len = 64, max tokens = 2, step joining = 0
    """

    seqs_max_tokens = [3, 2]
    prompts_lengths = [64, 64]
    steps_add_reqs = [0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequence 0
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 2,
            "n_reserved_blocks": 0,
        },
        {
            # Decode sequence 0
            "step": 3,
            "tkv": 66,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 1
            "step": 4,
            "tkv": 64,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_used_blocks": 1,
            "n_reserved_blocks": 1,
        },
        {
            # Decode sequence 1
            # Sequence 1 finishes at step 4
            # total blocks in use: 0
            "step": 5,
            "tkv": 65,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 6,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
            "n_reserved_blocks": 0,
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
    )
