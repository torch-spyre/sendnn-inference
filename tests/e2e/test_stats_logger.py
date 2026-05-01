import json
from pathlib import Path

import pytest
from spyre_util import ModelInfo, get_chicken_soup_prompts
from vllm import LLM

from sendnn_inference import envs as envs_spyre


@pytest.mark.cpu
def test_file_stats_logger(
    model: ModelInfo, max_model_len, max_num_seqs, max_num_batched_tokens, tmp_path
):
    prompts = get_chicken_soup_prompts(4)

    envs_spyre.override("SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED", "1")
    envs_spyre.override("SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR", str(tmp_path))
    envs_spyre.override("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager")

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        disable_log_stats=False,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    spyre_model.generate(prompts=prompts)

    assert Path(tmp_path / "request_metrics.jsonl").exists()

    with Path(tmp_path / "request_metrics.jsonl").open() as f:
        for line in f.readlines():
            data = json.loads(line)
            assert "prefill_interrupt_seconds" in data
            assert "e2e_latency_seconds" in data
            assert "timestamp" in data
