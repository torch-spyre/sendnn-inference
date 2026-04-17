import inspect

import pytest
from unittest.mock import Mock
from vllm.config import (
    VllmConfig,
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig,
    ProfilerConfig,
)
from vllm.v1.worker.gpu_worker import Worker as UpstreamWorker
from vllm_spyre.v1.worker.spyre_worker import SpyreWorker


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig with profiler enabled."""
    model_config = Mock(spec=ModelConfig)
    model_config.model = "test-model"
    model_config.runner_type = "generate"
    model_config.seed = 0
    model_config.max_model_len = 2048

    cache_config = Mock(spec=CacheConfig)

    parallel_config = Mock(spec=ParallelConfig)
    parallel_config.world_size = 1
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1

    scheduler_config = Mock(spec=SchedulerConfig)
    scheduler_config.max_num_seqs = 256

    vllm_config = Mock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.parallel_config = parallel_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.profiler_config = ProfilerConfig(
        profiler="torch", torch_profiler_dir="/tmp/test_traces"
    )
    vllm_config.instance_id = "test-instance"

    return vllm_config


def _make_worker(monkeypatch, vllm_config):
    monkeypatch.setattr("vllm_spyre.v1.worker.spyre_worker.ChunkedPrefillModelRunner", Mock())
    monkeypatch.setattr("vllm_spyre.v1.worker.spyre_worker.perf_metrics", Mock())
    return SpyreWorker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method="env://",
        is_driver_worker=True,
    )


@pytest.mark.cpu
@pytest.mark.worker
def test_profile_raises_when_profiler_not_enabled(monkeypatch, mock_vllm_config):
    """Test that profile() raises RuntimeError when profiler is not enabled."""
    mock_vllm_config.profiler_config = ProfilerConfig(profiler=None)
    worker = _make_worker(monkeypatch, mock_vllm_config)

    assert worker.profiler is None

    with pytest.raises(RuntimeError, match="Profiling is not enabled"):
        worker.profile(is_start=True)


def test_profile_signature_matches_upstream():
    """Test that SpyreWorker.profile() signature matches the upstream vllm Worker."""
    spyre_params = set(inspect.signature(SpyreWorker.profile).parameters)
    upstream_params = set(inspect.signature(UpstreamWorker.profile).parameters)
    assert spyre_params == upstream_params


# Made with Bob
