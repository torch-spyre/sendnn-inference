"""
Tests cover:
  - _resolve_mm_utils_cls: exact registry match, model_type scan fallback, unknown raises
  - VisionEncoderRunner.__init__: local path → model_path kwarg, HF ID → variant kwarg
  - VisionEncoderRunner.execute_model: dtype cast, CPU output
  - encoder_process_main: READY signal, ERROR signal, job processed, stop_event, encode error
"""

import multiprocessing
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_vllm_config(model="fake-model"):
    cfg = MagicMock()
    cfg.model_config.model = model
    cfg.model_config.revision = None
    return cfg


def _make_mm_encode_request(req_id="req-1"):
    from sendnn_inference.v1.core.scheduler import MMEncodeRequest

    return MMEncodeRequest(request_id=req_id, prompt_token_ids=[1, 2, 3], mm_features=[])


# ---------------------------------------------------------------------------
# _resolve_mm_utils_cls
# ---------------------------------------------------------------------------


class TestResolveMmUtilsCls:
    def test_model_type_scan_fallback_for_base_pretrained_config(self):
        """When hf_config is a base PretrainedConfig (Pydantic serialization loss),
        the fallback scan by model_type string must still find the right class."""
        from transformers import PretrainedConfig

        from sendnn_inference.v1.worker.mm_encoder_process import _resolve_mm_utils_cls

        class KnownHfConfig(PretrainedConfig):
            model_type = "known_scan"

        KnownUtils = type("KnownUtils", (), {})

        # base PretrainedConfig carrying model_type only (class info lost)
        base_cfg = PretrainedConfig()
        base_cfg.model_type = "known_scan"

        with patch("sendnn_inference.multimodal.MM_HF_CFG_REGISTRY", {KnownHfConfig: KnownUtils}):
            result = _resolve_mm_utils_cls(base_cfg)

        assert result is KnownUtils

    def test_unknown_model_type_raises_value_error(self):
        from transformers import PretrainedConfig

        from sendnn_inference.v1.worker.mm_encoder_process import _resolve_mm_utils_cls

        unknown_cfg = PretrainedConfig()
        unknown_cfg.model_type = "totally_unknown_xyz"

        with (
            patch("sendnn_inference.multimodal.MM_HF_CFG_REGISTRY", {}),
            pytest.raises(ValueError, match="no MMUtils found"),
        ):
            _resolve_mm_utils_cls(unknown_cfg)


# ---------------------------------------------------------------------------
# VisionEncoderRunner — __init__ (model loading path selection)
# ---------------------------------------------------------------------------


def _make_runner(vllm_config):
    """Construct a VisionEncoderRunner with all heavy deps mocked.
    Returns (runner, get_model_mock)."""
    from sendnn_inference.platform import SpyrePlatform
    from sendnn_inference.v1.worker.mm_encoder_process import VisionEncoderRunner

    fake_fms_model = MagicMock()
    fake_utils_cls = MagicMock()
    fake_utils_cls.__name__ = "FakeUtils"
    fake_utils_cls.mm_parameter_prefixes = ("vision_tower.",)
    get_model_mock = MagicMock(return_value=fake_fms_model)

    with (
        patch("fms.models.get_model", get_model_mock),
        patch.object(SpyrePlatform, "maybe_ensure_sendnn_configured"),
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process.SpyreCausalLM.resolve_hf_config",
            return_value=MagicMock(),
        ),
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process._resolve_mm_utils_cls",
            return_value=fake_utils_cls,
        ),
        patch(
            "sendnn_inference.v1.worker.mm_encoder_process.cast_params_for_spyre",
            return_value="cpu",
        ),
    ):
        runner = VisionEncoderRunner(vllm_config)

    return runner, get_model_mock


class TestVisionEncoderRunnerInit:
    def test_local_path_passes_model_path_kwarg(self, tmp_path):
        """When model is a local directory, get_model must receive model_path=."""
        cfg = _make_vllm_config(model=str(tmp_path))
        _, get_model_mock = _make_runner(cfg)

        _, kwargs = get_model_mock.call_args
        assert "model_path" in kwargs
        assert kwargs["model_path"] == str(tmp_path)
        assert "variant" not in kwargs

    def test_hf_id_passes_variant_kwarg(self):
        """When model is a non-local HF ID, get_model must receive variant=."""
        cfg = _make_vllm_config(model="org/model-name")

        with patch("os.path.isdir", return_value=False):
            _, get_model_mock = _make_runner(cfg)

        _, kwargs = get_model_mock.call_args
        assert "variant" in kwargs
        assert kwargs["variant"] == "org/model-name"
        assert "model_path" not in kwargs

    def test_vision_only_and_fused_weights_always_set(self, tmp_path):
        """vision_only=True and fused_weights=False must always be passed."""
        cfg = _make_vllm_config(model=str(tmp_path))
        _, get_model_mock = _make_runner(cfg)

        _, kwargs = get_model_mock.call_args
        assert kwargs.get("vision_only") is True
        assert kwargs.get("fused_weights") is False


# ---------------------------------------------------------------------------
# VisionEncoderRunner — execute_model
# ---------------------------------------------------------------------------


class TestVisionEncoderRunnerExecuteModel:
    def _make_runner_direct(self):
        """Build a VisionEncoderRunner instance bypassing __init__."""
        from sendnn_inference.v1.worker.mm_encoder_process import VisionEncoderRunner

        runner = VisionEncoderRunner.__new__(VisionEncoderRunner)
        runner._decoder_dtype = torch.float16
        runner.mm_device = "cpu"
        runner.mm_utils_cls = MagicMock()
        runner.fms_model = MagicMock()
        return runner

    def test_output_is_float16_cpu_contiguous(self):
        """execute_model must cast to _decoder_dtype and return a CPU tensor."""
        runner = self._make_runner_direct()
        # Simulate vision encoder returning float32
        raw_embeds = torch.ones(1, 8, 16, dtype=torch.float32)
        runner.mm_utils_cls.get_maybe_mm_embeddings.return_value = raw_embeds

        job = _make_mm_encode_request()
        result = runner.execute_model(job)

        assert result.dtype == torch.float16
        assert result.device.type == "cpu"
        assert result.is_contiguous()

    def test_input_ids_built_from_prompt_token_ids(self):
        """execute_model must pass the job's prompt_token_ids as input_ids."""
        runner = self._make_runner_direct()
        runner.mm_utils_cls.get_maybe_mm_embeddings.return_value = torch.zeros(
            1, 4, 8, dtype=torch.float16
        )
        job = _make_mm_encode_request()
        job.prompt_token_ids = [10, 20, 30]

        runner.execute_model(job)

        call_kwargs = runner.mm_utils_cls.get_maybe_mm_embeddings.call_args
        input_ids = call_kwargs[0][1]  # second positional arg
        assert input_ids.shape == (1, 3)
        assert input_ids.tolist() == [[10, 20, 30]]


# ---------------------------------------------------------------------------
# encoder_process_main
# ---------------------------------------------------------------------------


class TestEncoderProcessMain:
    """Tests for the encoder subprocess entry point.

    encoder_process_main is called directly (not via subprocess) with mocked
    VisionEncoderRunner so no model weights are loaded.
    """

    def test_ready_signal_on_success(self):
        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        jq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        stop = multiprocessing.Event()
        jq.put(None)  # sentinel → loop exits immediately after READY

        with patch("sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner"):
            encoder_process_main(_make_vllm_config(), jq, rq, stop)

        assert rq.get(timeout=2) == "READY"
        assert rq.empty()

    def test_error_signal_on_load_failure(self):
        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        jq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        stop = multiprocessing.Event()

        with patch(
            "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
            side_effect=RuntimeError("load failed"),
        ):
            encoder_process_main(_make_vllm_config(), jq, rq, stop)

        # multiprocessing.Queue uses a background writer thread; use a short
        # timeout instead of get_nowait() to avoid a race on early-exit paths.
        signal = rq.get(timeout=2)
        assert signal.startswith("ERROR:")
        assert "load failed" in signal

    def test_job_processed_result_on_queue(self):
        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        jq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        stop = multiprocessing.Event()

        fake_embeds = torch.zeros(1, 4, 8, dtype=torch.float16)
        job = _make_mm_encode_request("req-job")
        jq.put(job)
        jq.put(None)  # terminate after job

        mock_runner = MagicMock()
        mock_runner.execute_model.return_value = fake_embeds
        fake_shm = MagicMock()

        with (
            patch(
                "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
                return_value=mock_runner,
            ),
            patch(
                "sendnn_inference.v1.worker.mm_encoder_process.write_embeddings",
                return_value=fake_shm,
            ),
        ):
            encoder_process_main(_make_vllm_config(), jq, rq, stop)

        assert rq.get(timeout=2) == "READY"
        req_id, shape, dtype = rq.get(timeout=2)
        assert req_id == "req-job"
        assert shape == tuple(fake_embeds.shape)
        assert dtype == fake_embeds.dtype

    def test_encode_failure_puts_none_metadata(self):
        """When execute_model raises, (req_id, None, None) must be put on result queue."""
        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        jq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        stop = multiprocessing.Event()

        job = _make_mm_encode_request("req-fail")
        jq.put(job)
        jq.put(None)

        mock_runner = MagicMock()
        mock_runner.execute_model.side_effect = RuntimeError("encode error")

        with patch(
            "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
            return_value=mock_runner,
        ):
            encoder_process_main(_make_vllm_config(), jq, rq, stop)

        assert rq.get(timeout=2) == "READY"
        req_id, shape, dtype = rq.get(timeout=2)
        assert req_id == "req-fail"
        assert shape is None
        assert dtype is None

    def test_cancel_queue_skips_job_before_encode(self):
        """When a req_id is on the cancel_queue before its job is dequeued,
        execute_model must not be called and (req_id, None, None) is returned."""
        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        jq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        cq = multiprocessing.Queue()
        stop = multiprocessing.Event()

        cq.put("req-cancel")  # cancel token arrives before the job
        job = _make_mm_encode_request("req-cancel")
        jq.put(job)
        jq.put(None)

        mock_runner = MagicMock()

        with (
            patch(
                "sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner",
                return_value=mock_runner,
            ),
            patch("sendnn_inference.v1.worker.mm_encoder_process.write_embeddings"),
        ):
            encoder_process_main(_make_vllm_config(), jq, rq, stop, cq)

        assert rq.get(timeout=2) == "READY"
        assert rq.get(timeout=2) == ("req-cancel", None, None)
        assert not mock_runner.execute_model.called

    def test_stop_event_terminates_loop(self):
        from sendnn_inference.v1.worker.mm_encoder_process import encoder_process_main

        jq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        stop = multiprocessing.Event()
        stop.set()  # pre-set: loop exits on first iteration check

        with patch("sendnn_inference.v1.worker.mm_encoder_process.VisionEncoderRunner"):
            encoder_process_main(_make_vllm_config(), jq, rq, stop)

        assert rq.get(timeout=2) == "READY"
        assert rq.empty()
