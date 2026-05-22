"""Verification of the holdback feature in the chunked prefill scheduler.

This tests the relaxed constraint checking where requests are scheduled if
prefill constraints are satisfied (not future constraints). Requests that
would violate constraints during decode will be held back at that time.

The two main constraints checked at prefill time are:
1. Max-context constraint: current tkv <= max_context_len
2. Volumetric constraint: current_max_tkv * batch_size <= max_batch_tkv_limit

Run `python -m pytest tests/e2e/test_spyre_holdback_scheduler_steps.py`.
"""

import pytest
from scheduling_utils import (
    validate_scheduler_steps,
    create_request_for_scheduler_test,
    random_prompt,
)
from spyre_util import ModelInfo


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_holdback_prefill_max_context_ok(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Test that requests are scheduled when prefill max-context constraint is satisfied.

    With holdback feature, we only check if current tkv <= max_context_len,
    not if future max_tokens would fit. This request would have been blocked
    in the old scheduler but can now be scheduled.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 20, step joining = 0
            * 1: len = 70, max tokens = 10, step joining = 0
    """

    requests = [
        create_request_for_scheduler_test(
            model=model,
            request_id=0,
            add_step=0,
            max_tokens=20,
            prompt=random_prompt(model, seed=0, length=49),
            use_golden_token_injection=False,
            generate_hf_results=True,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=1,
            add_step=0,
            max_tokens=10,
            prompt=random_prompt(model, seed=1, length=70),
            use_golden_token_injection=False,
            generate_hf_results=True,
        ),
    ]

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
            "tkv": 49,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 0
            "step": 2,
            "tkv": 50,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # With holdback: sequence 1 CAN be scheduled now because
            # prefill tkv constraint is satisfied (70 <= 128)
            # Old scheduler would block this because future max_tokens
            # would exceed max_context_len (70 + 98 > 168)
            "step": 3,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 3,
        },
        {
            # Both sequences decode
            "step": 4,
            "tkv": 115,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 3,
        },
        {
            # Sequence 1 finishes
            "step": 12,
            "tkv": 123,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 0
            # We removed the padding block induced by sequence 1
            "step": 13,
            "tkv": 60,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 0
            # tkv is expanding to new block
            "step": 18,
            "tkv": 65,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 2,
        },
        {
            # Sequence 0 finishes
            "step": 21,
            "tkv": 68,
            "waiting": [],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_used_blocks": 0,
        },
        {
            # tkv should be cleared one step later
            "step": 22,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "finished_requests": [],
            "n_used_blocks": 0,
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=requests,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
    )


# NOTE Keeping the test draft just in case, but probably we won't need it at all
# and can remove the constraint along with the tests
# @pytest.mark.chunked_prefill
# @pytest.mark.full_model
# @pytest.mark.parametrize("max_num_seqs", [2])
# @pytest.mark.parametrize("max_model_len", [128])
# @pytest.mark.parametrize("max_num_batched_tokens", [128])
# @pytest.mark.parametrize("available_blocks", [None])
# def test_holdback_prefill_max_context_violated(
#     model: ModelInfo,
#     backend: str,
#     monkeypatch: pytest.MonkeyPatch,
#     set_random_seed,
#     max_num_seqs: int,
#     max_model_len: int,
#     max_num_batched_tokens: int,
#     available_blocks: int,
# ):
#     """Test that requests are blocked when prefill max-context constraint is violated.

#     Even with holdback, if the prefill tkv would exceed max_context_len,
#     the request cannot be scheduled.

#     Configuration:
#         * max_num_seqs: 2
#         * number of prompts: 2
#             * 0: len = 60, max tokens = 10, step joining = 0
#             * 1: len = 75, max tokens = 5, step joining = 0 (exceeds max_model_len)
#     """

#     requests = [
#         create_request_for_scheduler_test(
#             model=model,
#             request_id=0,
#             add_step=0,
#             max_tokens=10,
#             prompt=random_prompt(model, seed=0, length=60),
#             use_golden_token_injection=False,
#             generate_hf_results=False,
#         ),
#         create_request_for_scheduler_test(
#             model=model,
#             request_id=1,
#             add_step=6,
#             max_tokens=10,
#             prompt=random_prompt(model, seed=1, length=75),
#             use_golden_token_injection=False,
#             generate_hf_results=False,
#         ),
#     ]

#     checked_steps = [
#         {
#             "step": 0,
#             "tkv": 0,
#             "waiting": ["0"],
#             "running": [],
#             "request_outputs": [],
#             "n_used_blocks": 0,
#         },
#         {
#             # Prefill sequence 0
#             "step": 1,
#             "tkv": 60,
#             "waiting": [],
#             "running": ["0"],
#             "request_outputs": ["0"],
#             "n_used_blocks": 1,
#         },
#         {
#             # Decode 1 sequence 0
#             "step": 2,
#             "tkv": 61,
#             "waiting": [],
#             "running": ["0"],
#             "request_outputs": ["0"],
#             "n_used_blocks": 1,
#         },
#         {
#             # Decode 5 sequence 0
#             # Request 1 joins the waiting queue
#             "step": 6,
#             "tkv": 65,
#             "waiting": ["1"],
#             "running": ["0"],
#             "request_outputs": ["0"],
#             "n_used_blocks": 2,
#         },
#         {
#             # Decode 6 sequence 0
#             # Sequence 1 CANNOT be scheduled because the padding-induced to
#             # request 0 would shift its tkv beyond max_context_len
#             # (64 + 66 = 130 > 128)
#             "step": 7,
#             "tkv": 75,
#             "waiting": [],
#             "running": ["1", "0"],
#             "request_outputs": ["0"],
#             "n_used_blocks": 2,
#         },
#         # {
#         #     # Decode 6 sequence 0
#         #     # Sequence 1 CANNOT be scheduled because the padding-induced to
#         #     # request 0 would shift its tkv beyond max_context_len
#         #     # (64 + 66 = 130 > 128)
#         #     "step": 7,
#         #     "tkv": 66,
#         #     "waiting": ["1"],
#         #     "running": ["0"],
#         #     "request_outputs": ["0"],
#         #     "n_used_blocks": 2,
#         # },
#         {
#             # Sequence 0 finishes
#             "step": 10,
#             "tkv": 70,
#             "waiting": ["1"],
#             "running": [],
#             "request_outputs": ["0"],
#             "finished_requests": ["0"],
#             "n_used_blocks": 2,
#         },
#         {
#             # Prefill sequence 1
#             "step": 11,
#             "tkv": 70,
#             "waiting": ["1"],
#             "running": [],
#             "request_outputs": ["0"],
#             "finished_requests": ["0"],
#             "n_used_blocks": 2,
#         },
#         # {
#         #     # Continue decoding sequence 0
#         #     "step": 3,
#         #     "tkv": 66,
#         #     "waiting": ["1"],
#         #     "running": ["0"],
#         #     "request_outputs": ["0"],
#         #     "n_used_blocks": 1,
#         # },
#     ]

#     validate_scheduler_steps(
#         model=model,
#         backend=backend,
#         monkeypatch=monkeypatch,
#         requests=requests,
#         checked_steps=checked_steps,
#         max_num_seqs=max_num_seqs,
#         max_model_len=max_model_len,
#         available_blocks=available_blocks,
#         max_num_batched_tokens=max_num_batched_tokens,
#     )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [4])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_holdback_prefill_volumetric_ok(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Test that requests are scheduled when prefill volumetric constraint is satisfied.

    With holdback, we only check current_max_tkv * batch_size <= limit,
    not future volumetric constraints.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 64, max tokens = 100, step joining = 0
            * 1: len = 65, max tokens = 100, step joining = 0
    """

    # Prefill volume: 2 * 65 = 130 (should pass)
    # Old scheduler would check future: 2 * (65 + 100) = 330 (would fail with limit 200)
    max_batch_tkv_limit = 256

    requests = [
        create_request_for_scheduler_test(
            model=model,
            request_id=0,
            add_step=0,
            max_tokens=100,
            prompt=random_prompt(model, seed=0, length=64),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=1,
            add_step=0,
            max_tokens=100,
            prompt=random_prompt(model, seed=1, length=65),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
    ]

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
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # With holdback: sequence 1 CAN be scheduled
            # Prefill volume: 2 * 65 = 130 <= 200 (passes)
            # Old scheduler would block: future volume 2 * 165 = 330 > 200
            "step": 2,
            "tkv": 65,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 3,
        },
        {
            # Both sequences decode
            "step": 3,
            "tkv": 66,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 3,
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=requests,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_batch_tkv_limit=max_batch_tkv_limit,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_holdback_prefill_volumetric_violated(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Test that requests are blocked when prefill volumetric constraint is violated.

    Even with holdback, if prefill volume exceeds limit, request cannot be scheduled.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 64, max tokens = 10, step joining = 0
            * 1: len = 65, max tokens = 10, step joining = 0
    """

    requests = [
        create_request_for_scheduler_test(
            model=model,
            request_id=0,
            add_step=0,
            max_tokens=10,
            prompt=random_prompt(model, seed=0, length=64),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=1,
            add_step=0,
            max_tokens=10,
            prompt=random_prompt(model, seed=1, length=65),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
    ]

    # Prefill volume would be: 2 * 65 = 130
    max_batch_tkv_limit = 129  # Just below the prefill requirement

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
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Sequence 1 CANNOT be scheduled
            # Prefill volume: 2 * 65 = 130 > 129 (limit)
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Sequence 0 continues decoding
            "step": 3,
            "tkv": 66,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=requests,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_batch_tkv_limit=max_batch_tkv_limit,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [3])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_holdback_prefill_multiple_requests_ok(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    available_blocks: int,
):
    """Test scheduling multiple requests with prefill constraint checking.

    All requests can be scheduled as long as prefill constraints are satisfied,
    regardless of future max_tokens requirements.

    Configuration:
        * max_num_seqs: 3
        * number of prompts: 3
            * 0: len = 50, max tokens = 200, step joining = 0
            * 1: len = 60, max tokens = 200, step joining = 0
            * 2: len = 70, max tokens = 200, step joining = 0
    """

    monkeypatch.setenv("SENDNN_INFERENCE_CP_INTERLEAVE_STEPS", "0")

    requests = [
        create_request_for_scheduler_test(
            model=model,
            request_id=0,
            add_step=0,
            max_tokens=200,
            prompt=random_prompt(model, seed=0, length=50),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=1,
            add_step=0,
            max_tokens=200,
            prompt=random_prompt(model, seed=1, length=60),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=2,
            add_step=0,
            max_tokens=200,
            prompt=random_prompt(model, seed=2, length=70),
            use_golden_token_injection=False,
            generate_hf_results=False,
        ),
    ]

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
            "tkv": 50,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # With holdback: can schedule even though future tokens would be large
            "step": 2,
            "tkv": 60,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 2
            # With holdback: can schedule even though future tokens would be large
            "step": 3,
            "tkv": 70,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2"],
            "n_used_blocks": 3,
        },
        {
            # All three sequences decode together
            "step": 4,
            "tkv": 71,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2", "1", "0"],
            "n_used_blocks": 3,
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=requests,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
    )
