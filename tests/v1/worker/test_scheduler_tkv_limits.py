import pytest
from scheduling_utils import create_request_for_scheduler_test, random_prompt
from v1.worker.mock_model import InstrumentedModelRunner
from spyre_util import REFERENCE_MODELS
from vllm_spyre.platform import SpyrePlatform

@pytest.mark.cpu
@pytest.mark.chunked_prefill
def test_batch_tkv_padding_alignment_bug(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DT_MAX_BATCH_TKV_LIMIT", "131072")

    model = REFERENCE_MODELS[InstrumentedModelRunner.DEFAULT_TEST_MODEL]

    model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=512,
        max_num_seqs=32,
        max_model_len=32768,
        available_blocks=32768,
    )
    
    req0_prompt = random_prompt(model=model, seed=0, length=940)
    req1_prompt = random_prompt(model=model, seed=1, length=412)
    req2_prompt = random_prompt(model=model, seed=2, length=969)
    req3_prompt = random_prompt(model=model, seed=2, length=949)
    req4_prompt = random_prompt(model=model, seed=2, length=11946)
    max_tokens = 16384
    req0 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=max_tokens,
        prompt=req0_prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    ).request

    req1 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=max_tokens,
        prompt=req1_prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    ).request

    req2 = create_request_for_scheduler_test(
        model=model,
        request_id=2,
        add_step=0,
        max_tokens=max_tokens,
        prompt=req2_prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    ).request

    req3 = create_request_for_scheduler_test(
        model=model,
        request_id=3,
        add_step=0,
        max_tokens=max_tokens,
        prompt=req3_prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    ).request

    req4 = create_request_for_scheduler_test(
        model=model,
        request_id=4,
        add_step=0,
        max_tokens=max_tokens,
        prompt=req4_prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
    ).request

    scheduler = model_runner.scheduler
    scheduler.max_batch_tkv_limit = 131072
    SpyrePlatform._max_batch_tkv_limit = 131072

    scheduler.add_request(req0)
    scheduler.add_request(req1)
    scheduler.add_request(req2)
    scheduler.add_request(req3)
    scheduler.add_request(req4)
    running = True
    while running:
        sched_output = scheduler.schedule()
        output = model_runner.execute_model(sched_output)
        scheduler.update_from_output(sched_output, output)
        running = len(scheduler.running) > 0

