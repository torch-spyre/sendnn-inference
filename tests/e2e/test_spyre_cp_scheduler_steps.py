"""Verification of the correctness of the step-by-step execution of chunked
prefill. It does so by comparing, at every engine step (i.e. prefill or decode
iteration), a bunch of attributes. This allows a finer testing of the padding
and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_cp_inference_steps.py`.
"""

import pytest
from scheduling_utils import check_scheduler_inference_steps
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
    """Scenario where the requested prompt is too long for current tkv value

    Note that as we could prefill the prompt straight away, however,
    in this test the max model length is decreased to a value where
    the tkv of the decode batch would be shifted beyond the max model length,
    we therefore have to wait with scheduling.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 17, step joining = 0
            * 1: len = 70, max tokens = 17, step joining = 0
    """

    seqs_max_tokens = [17, 17]
    prompts_lengths = [49, 70]
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
            "tkv": 49,  # prompt len
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        # Here we cannot schedule sequence 1. By shifting sequence 0 by
        # 1 block its max tkv would exceed the max model length
        {
            # Decode sequence 0
            # total blocks in use: 1 (writing into right pads)
            "step": 2,
            "tkv": 50,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1, tkv large enough to prefill w/o tkv shift
            # total blocks in use: 1 + 2
            "step": 17,
            # add 64 to tkv of seq 0 (64) to have it in the same block as seq 1
            "tkv": 128,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 2 + 2 (prefill (2 block) + 17 decodes in the last block)
            "n_used_blocks": 3,
        },
        {
            # Decode sequences 0 and 1
            # Sequence 0 finishes
            "step": 18,
            # remove left padding of seq 0, and keep its tkv in the same block
            # as seq 1: 129 - 64 = 65
            # tkv of seq 1 is now max
            "tkv": 71,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_used_blocks": 2,  # seq 0 needs another block for the last token
        },
        {
            # Decode sequence 1
            # total blocks in use: 4 - 2 = 2
            "step": 19,
            "tkv": 72,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_used_blocks": 2,
        },
        {
            # Sequence 1 finishes at step 33
            "step": 33,
            "tkv": 86,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_used_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 34,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
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
        },
        {
            # Prefill sequence 0 chunk 1
            "step": 2,
            "tkv": 512,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_used_blocks": 4,
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
        },
        {
            # Tkv should be cleared one step later
            "step": 6,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
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
        },
        {
            # Chunk 0 of request 1 prefill
            "step": 3,
            "tkv": 11,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_used_blocks": 3,
        },
        {
            # Decode 2 of request 0.
            "step": 4,
            "tkv": 12,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 3,
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
        },
        {
            # Decode 3 of request 0.
            "step": 6,
            "tkv": 13,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 5,
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
        },
        {
            # Decode 4 of request 0.
            "step": 8,
            "tkv": 14,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 7,
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
        },
        {
            # Tkv should be cleared one step later
            "step": 13,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
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
        },
        {
            # Both requests start decoding.
            "step": 6,
            "tkv": 523,  # tkv jump
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
        },
        {
            # Decode 2
            "step": 7,
            "tkv": 524,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 10,
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
        },
        {
            # Decode 4 of request 0.
            "step": 9,
            "tkv": 14,  # tkv jump
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode 5 of request 0.
            "step": 10,
            "tkv": 15,  # tkv jump
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode 6 of request 0.
            "step": 11,
            "tkv": 16,  # tkv jump
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
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
        },
        {
            # Tkv should be cleared one step later
            "step": 13,  # with or without interleaving we finish at step 13
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
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
        },
        {
            # Decode 1 of request 0.
            "step": 2,
            "tkv": 11,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode 2 of request 0.
            "step": 3,
            "tkv": 12,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
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
        },
        {
            # Decode 3 of request 0.
            "step": 5,
            "tkv": 13,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 3,
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
        },
        {
            # Decode 4 of request 0.
            "step": 7,
            "tkv": 14,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 5,
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
        },
        {
            # Decode 5 of request 0.
            "step": 9,
            "tkv": 15,
            "waiting": [],
            "running": ["0", "1"],
            "request_outputs": ["0"],
            "n_used_blocks": 7,
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
        },
        {
            "step": 13,
            "tkv": 515,  # tkv jump
            "waiting": [],
            "running": [],
            "finished_requests": ["1"],
            "request_outputs": ["1"],
            "n_used_blocks": 0,
        },
        {
            # Tkv should be cleared one step later
            "step": 14,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
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


# TODO had to move test at the end, having it after test_prefill_tkv_too_big
# was breaking the ordering ("error in test ordering!")
# looks like an issue with sorting the runtime configurations
@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [4])
@pytest.mark.parametrize("max_model_len", [128])  # restricted to violate scheduler condition
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefill_tkv_too_big2(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Scenario where the requested number of output is too big for current
    tkv value. We need to wait for a previous long prompt request to finish and
    have tkv reduced to a previous block before being able to schedule the
    new request.

    Configuration:
        * max_num_seqs: 4
        * number of prompts: 3
            * 0: len = 20, max tokens = 5, step joining = 0
            * 1: len = 80, max tokens = 3, step joining = 0
            * 2: len = 16, max tokens = 50, step joining = 0
    """

    monkeypatch.setenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "0")

    seqs_max_tokens = [5, 3, 50]
    prompts_lengths = [20, 80, 16]
    steps_add_reqs = [0, 0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            "step": 1,
            "tkv": 20,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        # tkv should be updated at the end of the last chunked prefill
        # here we have only one chunk, so it will be updated directly
        {
            # Prefill sequence 1
            "step": 2,
            "tkv": 84,  # 64 (1 block padding) + 20 (prompt of seq 0) = 84
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 1 + 2 (prefill (2 block) + 3 decodes in the last block)
            "n_used_blocks": 3,
        },
        # Here we cannot schedule sequence 2. Current tkv being in the second
        # block, the number of requested tokens can't fit in the remaining space
        # (64 (full block left padding) + 16 (prompt) + 50 (decode) = 130 > 128)
        {
            # Decode 1 of sequence 0
            # Decode 1 of sequence 1
            "step": 3,
            "tkv": 85,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 3,
        },
        {
            # Decode 2 of sequence 0
            # Decode 2 of sequence 1
            # Sequence 1 finishes
            "step": 4,
            "tkv": 86,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_used_blocks": 1,
        },
        # The tkv value used here is computed before the model forward pass and
        # token sampling of this step. As a result, it does not yet reflect
        # sequences that finish in the current step. In this case, tkv=86 still
        # includes sequence 1, which completes in this step, and this will only
        # be accounted for in the next step. Therefore, sequence 2 cannot be
        # prefilled yet at this point.
        {
            # Decode 3 of sequence 0
            "step": 5,
            "tkv": 23,  # 20 (prompt len) + 3 (decodes) = 23
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        # Sequence 2 can be scheduled for prefill, now that tkv is moved back to
        # the first block.
        {
            # Prefill sequence 2
            "step": 6,
            "tkv": 23,
            "waiting": [],
            "running": ["2", "0"],
            "request_outputs": ["2"],
            # 3 - 2 (finished seq 1) + 2 (prefill + 50 decodes in new block)
            "n_used_blocks": 2,
        },
        {
            # Decode 4 of sequence 0
            # Decode 1 of sequence 2
            # Sequence 0 finishes
            "step": 7,
            "tkv": 24,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2", "0"],
            "finished_requests": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode 2 of sequence 2
            "step": 8,
            "tkv": 18,  # 16 (prompt len) + 2 (decodes) = 18
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_used_blocks": 1,
        },
        {
            # Decode 49 of sequence 2
            # Sequence 2 finishes
            "step": 55,
            "tkv": 65,  # 16 (prompt len) + 49 (decodes) = 65
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_used_blocks": 0,
        },
        {
            # tkv should be cleared one step later
            "step": 56,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_used_blocks": 0,
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
