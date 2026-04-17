import pytest
from vllm import LLM
from vllm.config import ProfilerConfig

from spyre_util import ModelInfo, get_chicken_soup_prompts
from vllm_spyre import envs as envs_spyre


@pytest.mark.cpu
def test_profiler(
    model: ModelInfo,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    tmp_path,
):
    prompts = get_chicken_soup_prompts(2)

    envs_spyre.override("VLLM_SPYRE_DYNAMO_BACKEND", "eager")

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
        ),
    )

    spyre_model.start_profile()
    spyre_model.generate(prompts=prompts)
    spyre_model.stop_profile()

    trace_files = list(tmp_path.glob("*.pt.trace.json*"))
    assert len(trace_files) > 0


@pytest.mark.cpu
def test_profiler_prefix(
    model: ModelInfo,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    tmp_path,
):
    """Test that profile_prefix is applied to each call."""
    prompts = get_chicken_soup_prompts(2)

    envs_spyre.override("VLLM_SPYRE_DYNAMO_BACKEND", "eager")

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
        ),
    )

    for prefix in ("first", "second"):
        spyre_model.start_profile(profile_prefix=prefix)
        spyre_model.generate(prompts=prompts)
        spyre_model.stop_profile()

    for prefix in ("first", "second"):
        trace_files = list(tmp_path.glob(f"{prefix}*.pt.trace.json*"))
        assert len(trace_files) > 0, f"No trace files found for prefix '{prefix}'"
