"""Unit tests for SpyrePlatform.get_cpu_count."""

import pytest
from unittest.mock import MagicMock, mock_open, patch
from sendnn_inference.platform import SpyrePlatform


pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def reset_spyre_config():
    original = SpyrePlatform._config
    SpyrePlatform._config = MagicMock()
    yield
    SpyrePlatform._config = original


class TestGetCpuCount:
    def test_env_override_takes_priority(self, monkeypatch):
        monkeypatch.setenv("SENDNN_INFERENCE_NUM_CPUS", "12")
        cpu_count, msg = SpyrePlatform.get_cpu_count()
        assert cpu_count == 12.0
        assert "SENDNN_INFERENCE_NUM_CPUS" in msg

    def test_cgroup_quota_used_when_available(self, monkeypatch):
        monkeypatch.delenv("SENDNN_INFERENCE_NUM_CPUS", raising=False)
        with patch("sendnn_inference.platform.open", mock_open(read_data="200000 100000")):
            cpu_count, msg = SpyrePlatform.get_cpu_count()
        assert cpu_count == 2.0
        assert "cgroup" in msg

    def test_cgroup_max_is_skipped(self, monkeypatch):
        monkeypatch.delenv("SENDNN_INFERENCE_NUM_CPUS", raising=False)
        with (
            patch("sendnn_inference.platform.open", mock_open(read_data="max 100000")),
            patch.dict("sys.modules", {"psutil": None}),
            patch("os.cpu_count", return_value=8),
        ):
            cpu_count, msg = SpyrePlatform.get_cpu_count()
        # cgroup says unlimited → should fall through to os.cpu_count
        assert cpu_count == 8.0

    def test_psutil_physical_cores_used_when_no_cgroup(self, monkeypatch):
        monkeypatch.delenv("SENDNN_INFERENCE_NUM_CPUS", raising=False)
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 6
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.dict("sys.modules", {"psutil": mock_psutil}),
        ):
            cpu_count, msg = SpyrePlatform.get_cpu_count()
        assert cpu_count == 6.0
        assert "physical" in msg
        mock_psutil.cpu_count.assert_called_once_with(logical=False)

    def test_os_cpu_count_fallback_when_psutil_missing(self, monkeypatch):
        monkeypatch.delenv("SENDNN_INFERENCE_NUM_CPUS", raising=False)
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.dict("sys.modules", {"psutil": None}),
            patch("os.cpu_count", return_value=4),
        ):
            cpu_count, msg = SpyrePlatform.get_cpu_count()
        assert cpu_count == 4.0
        assert "os.cpu_count" in msg

    def test_returns_none_when_all_detection_fails(self, monkeypatch):
        monkeypatch.delenv("SENDNN_INFERENCE_NUM_CPUS", raising=False)
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.dict("sys.modules", {"psutil": None}),
            patch("os.cpu_count", return_value=None),
        ):
            cpu_count, msg = SpyrePlatform.get_cpu_count()
        assert cpu_count is None
