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
@pytest.mark.parametrize("max_num_seqs", [4])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [256])
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
            * 0: len = 15, max tokens = 60, step joining = 0
            * 1: len = 15, max tokens = 60, step joining = 0
            * 1: len = 66, max tokens = 60, step joining = 0
    """

    # Volume right after prefill: 3 * 82 = 246 (should pass)
    # Old scheduler would check future: TODO (would fail)
    max_batch_tkv_limit = 256

    requests = [
        create_request_for_scheduler_test(
            model=model,
            request_id=0,
            add_step=0,
            max_tokens=60,
            prompt=random_prompt(model, seed=0, length=15),
            use_golden_token_injection=False,
            generate_hf_results=True,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=1,
            add_step=0,
            max_tokens=60,
            prompt=random_prompt(model, seed=1, length=15),
            use_golden_token_injection=False,
            generate_hf_results=True,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=2,
            add_step=0,
            max_tokens=60,
            prompt=random_prompt(model, seed=2, length=66),
            use_golden_token_injection=False,
            generate_hf_results=True,
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
            "tkv": 15,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 0
            "step": 2,
            "tkv": 16,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            "step": 3,
            "tkv": 15,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            "step": 4,
            "tkv": 17,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 2
            # With holdback: sequence 2 CAN be scheduled
            # Decode volume: 3 * 82 = 246 <= 256 (passes)
            # Old scheduler would block: future volume 3 * TODO = 330 > 256
            "step": 5,
            "tkv": 66,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2"],
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0, 1, and 2
            "step": 6,
            "tkv": 82,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2", "1", "0"],
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0, 1, and 2
            "step": 9,
            "tkv": 85,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2", "1", "0"],
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0, 1, and 2
            # About to violate the volumetric constraint: 3 * 86 = 258 > 256
            # Holdback activates TODO
            "step": 10,
            "tkv": 86,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2", "1", "0"],
            "n_used_blocks": 4,
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
            * 0: len = 25, max tokens = 7, step joining = 0
            * 1: len = 25, max tokens = 6, step joining = 0
            * 1: len = 66, max tokens = 3, step joining = 0
    """

    # Volume right after prefill: 3 * 89 = 267 (fails)
    max_batch_tkv_limit = 256

    requests = [
        create_request_for_scheduler_test(
            model=model,
            request_id=0,
            add_step=0,
            max_tokens=7,
            prompt=random_prompt(model, seed=0, length=25),
            use_golden_token_injection=False,
            generate_hf_results=True,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=1,
            add_step=0,
            max_tokens=6,
            prompt=random_prompt(model, seed=1, length=25),
            use_golden_token_injection=False,
            generate_hf_results=True,
        ),
        create_request_for_scheduler_test(
            model=model,
            request_id=2,
            add_step=0,
            max_tokens=3,
            prompt=random_prompt(model, seed=2, length=66),
            use_golden_token_injection=False,
            generate_hf_results=True,
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
            "tkv": 25,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 0
            "step": 2,
            "tkv": 26,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            "step": 3,
            "tkv": 25,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            "step": 4,
            "tkv": 27,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            # Cannot prefill sequence 2
            # tkv would be 3 * (28 + 64) = 276 > 256
            "step": 5,
            "tkv": 28,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_used_blocks": 2,
        },
        {
            # Sequences 0 and 1 both finish
            "step": 8,
            "tkv": 31,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 2
            "step": 9,
            "tkv": 66,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_used_blocks": 2,
        },
        {
            # Decode sequence 2
            "step": 10,
            "tkv": 67,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_used_blocks": 2,
        },
        {
            # Sequence 2 finishes
            "step": 11,
            "tkv": 68,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_used_blocks": 0,
        },
        {
            # tkv should be cleared one step later
            "step": 12,
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
        max_batch_tkv_limit=max_batch_tkv_limit,
        max_num_batched_tokens=max_num_batched_tokens,
    )
