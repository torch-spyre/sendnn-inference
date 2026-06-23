import pytest
import torch
from multiprocessing.shared_memory import SharedMemory

from sendnn_inference.v1.worker.mm_shared_memory import (
    _shm_name,
    cleanup_embeddings,
    read_embeddings,
    write_embeddings,
)

pytestmark = [pytest.mark.multimodal, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_tensor(shape, dtype):
    """Create a distinct, non-trivial CPU tensor for round-trip checks."""
    t = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape)
    return t.to(dtype)


def _cleanup_if_exists(req_id: str) -> None:
    """Remove any leftover SHM segment from a previous (crashed) run."""
    try:
        shm = SharedMemory(name=_shm_name(req_id))
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# shm name helper
# ---------------------------------------------------------------------------


class TestShmName:
    def test_strips_dashes(self):
        name = _shm_name("550e8400-e29b-41d4-a716")
        assert "-" not in name

    def test_length_within_posix_limit(self):
        # POSIX requires name <= 255 chars; macOS requires <= 30 chars for /name
        name = _shm_name("a" * 100)
        assert len(name) <= 30

    def test_deterministic(self):
        assert _shm_name("req-abc") == _shm_name("req-abc")

    def test_different_ids_give_different_names(self):
        assert _shm_name("req-001") != _shm_name("req-002")


# ---------------------------------------------------------------------------
# write / read round-trip
# ---------------------------------------------------------------------------


class TestWriteReadRoundtrip:
    """End-to-end: write_embeddings → read_embeddings produces identical data."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        req_id = "test-rw-roundtrip"
        _cleanup_if_exists(req_id)
        yield req_id
        _cleanup_if_exists(req_id)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_roundtrip_all_dtypes(self, cleanup, dtype):
        req_id = f"test-rw-{dtype}"
        _cleanup_if_exists(req_id)

        shape = (1, 8, 16)
        original = _make_tensor(shape, dtype)

        shm = write_embeddings(original, req_id)
        try:
            result = read_embeddings(req_id, shape, dtype)
        finally:
            cleanup_embeddings(shm)

        assert result.shape == original.shape
        assert result.dtype == original.dtype
        assert torch.equal(result, original), f"Round-trip failed for dtype={dtype}"

        _cleanup_if_exists(req_id)

    def test_roundtrip_float16_large(self, cleanup):
        """Larger tensor (closer to real MM embedding size)."""
        req_id = "test-rw-large"
        _cleanup_if_exists(req_id)
        shape = (1, 1024, 1024)  # scaled-down embedding
        original = _make_tensor(shape, torch.float16)

        shm = write_embeddings(original, req_id)
        try:
            result = read_embeddings(req_id, shape, torch.float16)
        finally:
            cleanup_embeddings(shm)

        assert torch.equal(result, original)
        _cleanup_if_exists(req_id)

    def test_result_is_independent_of_shm(self, cleanup):
        """clone() in read_embeddings must detach the tensor from the SHM buffer."""
        req_id = "test-rw-independence"
        _cleanup_if_exists(req_id)
        shape = (1, 4, 8)
        original = _make_tensor(shape, torch.float32)

        shm = write_embeddings(original, req_id)
        result = read_embeddings(req_id, shape, torch.float32)
        cleanup_embeddings(shm)  # SHM is gone after this

        # Accessing result must not raise or return garbage
        assert result.shape == shape
        assert not torch.isnan(result).any()
        _cleanup_if_exists(req_id)


# ---------------------------------------------------------------------------
# write_embeddings: input handling
# ---------------------------------------------------------------------------


class TestWriteEmbeddings:
    @pytest.fixture(autouse=True)
    def req_id(self):
        rid = "test-write"
        _cleanup_if_exists(rid)
        yield rid
        _cleanup_if_exists(rid)

    def test_returns_shared_memory_handle(self, req_id):
        t = _make_tensor((1, 4, 8), torch.float16)
        shm = write_embeddings(t, req_id)
        assert isinstance(shm, SharedMemory)
        cleanup_embeddings(shm)

    def test_shm_size_matches_tensor_nbytes(self, req_id):
        t = _make_tensor((1, 4, 8), torch.float32)
        shm = write_embeddings(t, req_id)
        assert shm.size >= t.nbytes
        cleanup_embeddings(shm)

    def test_non_contiguous_tensor_is_handled(self, req_id):
        """Non-contiguous tensor (e.g. from a slice) should be made contiguous."""
        base = torch.arange(64, dtype=torch.float16).reshape(1, 8, 8)
        non_contig = base[:, ::2, :]  # every other row → non-contiguous
        assert not non_contig.is_contiguous()

        # Should not raise
        shm = write_embeddings(non_contig, req_id)
        shape = non_contig.shape
        result = read_embeddings(req_id, shape, torch.float16)
        cleanup_embeddings(shm)

        expected = non_contig.contiguous()
        assert torch.equal(result, expected)

    def test_unsupported_dtype_raises(self, req_id):
        t = torch.zeros((1, 4, 8), dtype=torch.int32)
        with pytest.raises(AssertionError):
            write_embeddings(t, req_id)

    def test_wrong_ndim_raises(self, req_id):
        t = torch.zeros((4, 8), dtype=torch.float16)  # 2D, not 3D
        with pytest.raises(AssertionError):
            write_embeddings(t, req_id)


# ---------------------------------------------------------------------------
# read_embeddings
# ---------------------------------------------------------------------------


class TestReadEmbeddings:
    def test_closes_handle_after_read(self):
        """SHM handle must be closed (but not unlinked) by read_embeddings."""
        req_id = "test-read-close"
        _cleanup_if_exists(req_id)
        shape = (1, 4, 8)
        t = _make_tensor(shape, torch.float16)

        shm = write_embeddings(t, req_id)
        result = read_embeddings(req_id, shape, torch.float16)

        # SHM should still exist (only write_embeddings' handle is open)
        # We can verify by opening it again
        shm2 = SharedMemory(name=_shm_name(req_id))
        shm2.close()

        cleanup_embeddings(shm)
        _cleanup_if_exists(req_id)

        assert result.shape == shape

    def test_correct_shape_and_dtype_returned(self):
        req_id = "test-read-shape"
        _cleanup_if_exists(req_id)
        shape = (1, 12, 32)
        t = _make_tensor(shape, torch.bfloat16)

        shm = write_embeddings(t, req_id)
        result = read_embeddings(req_id, shape, torch.bfloat16)
        cleanup_embeddings(shm)
        _cleanup_if_exists(req_id)

        assert result.shape == torch.Size(shape)
        assert result.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# cleanup_embeddings
# ---------------------------------------------------------------------------


class TestCleanupEmbeddings:
    def test_unlinks_segment(self):
        req_id = "test-cleanup-unlink"
        _cleanup_if_exists(req_id)
        t = _make_tensor((1, 4, 8), torch.float16)

        shm = write_embeddings(t, req_id)
        cleanup_embeddings(shm)

        # After cleanup, the name must no longer be accessible
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=_shm_name(req_id))

    def test_idempotent_on_double_call(self):
        """Calling cleanup_embeddings twice must not raise."""
        req_id = "test-cleanup-double"
        _cleanup_if_exists(req_id)
        t = _make_tensor((1, 4, 8), torch.float16)

        shm = write_embeddings(t, req_id)
        cleanup_embeddings(shm)
        cleanup_embeddings(shm)  # second call must be a silent no-op

    def test_stale_shm_does_not_leak(self):
        """If a process crashed leaving SHM behind, cleanup handles it."""
        req_id = "test-cleanup-stale"
        _cleanup_if_exists(req_id)

        # Simulate stale segment: create but don't clean up
        t = _make_tensor((1, 4, 8), torch.float32)
        shm = write_embeddings(t, req_id)
        shm.close()  # close handle without unlinking → stale segment

        # Second write to the same req_id will fail with FileExistsError.
        # The calling code (spyre_model_runner) is responsible for cleaning up
        # before re-creating. Verify that cleanup works on the stale handle.
        shm2 = SharedMemory(name=_shm_name(req_id))  # open stale segment
        cleanup_embeddings(shm2)

        # Now the segment is gone
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=_shm_name(req_id))


# ---------------------------------------------------------------------------
# data integrity via torch.frombuffer (zero-copy write path)
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    """Verify that torch.frombuffer write path preserves exact bit patterns."""

    def test_bfloat16_bit_pattern_preserved(self):
        req_id = "test-bits-bf16"
        _cleanup_if_exists(req_id)
        shape = (1, 6, 4)
        # Use specific values that have known bfloat16 representations
        t = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.5, 0.25, 0.125, 0.0625],
                    [100.0, 200.0, -1.0, -2.0],
                    [0.0, -0.0, float("inf"), -float("inf")],
                    [1e-3, 1e3, 1.23, -4.56],
                    [7.0, 8.0, 9.0, 10.0],
                ]
            ],
            dtype=torch.bfloat16,
        )

        shm = write_embeddings(t, req_id)
        result = read_embeddings(req_id, shape, torch.bfloat16)
        cleanup_embeddings(shm)
        _cleanup_if_exists(req_id)

        # View as int16 to compare raw bits (avoids NaN != NaN issues)
        assert torch.equal(t.view(torch.int16), result.view(torch.int16))

    def test_float16_special_values(self):
        req_id = "test-bits-f16"
        _cleanup_if_exists(req_id)
        shape = (1, 2, 4)
        t = torch.tensor(
            [[[0.0, -0.0, 1.0, -1.0], [65504.0, -65504.0, 1e-4, -1e-4]]], dtype=torch.float16
        )

        shm = write_embeddings(t, req_id)
        result = read_embeddings(req_id, shape, torch.float16)
        cleanup_embeddings(shm)
        _cleanup_if_exists(req_id)

        assert torch.equal(t.view(torch.int16), result.view(torch.int16))

    def test_all_zeros_preserved(self):
        req_id = "test-zeros"
        _cleanup_if_exists(req_id)
        shape = (1, 8, 8)
        t = torch.zeros(shape, dtype=torch.float32)

        shm = write_embeddings(t, req_id)
        result = read_embeddings(req_id, shape, torch.float32)
        cleanup_embeddings(shm)
        _cleanup_if_exists(req_id)

        assert torch.equal(result, t)


# ---------------------------------------------------------------------------
# store_mm_embeddings: aborted-request guard
# ---------------------------------------------------------------------------


class TestStoreMmEmbeddings:
    """Tests for ChunkedPrefillModelRunner.store_mm_embeddings."""

    def _make_runner(self):
        """Build a ChunkedPrefillModelRunner instance bypassing __init__."""
        from sendnn_inference.v1.worker.spyre_model_runner import ChunkedPrefillModelRunner

        runner = ChunkedPrefillModelRunner.__new__(ChunkedPrefillModelRunner)
        runner.rank = 0
        runner.requests = {}
        runner.pending_mm_embeddings = {}
        return runner

    def test_discards_embedding_for_aborted_request(self):
        """Embedding for a request no longer in self.requests must be discarded.

        If a request is aborted while its encode job is in-flight, the result
        arrives after _update_batch already removed it from self.requests.
        store_mm_embeddings must not write to pending_mm_embeddings — otherwise
        the tensor leaks indefinitely.
        """
        runner = self._make_runner()
        runner.requests = {}  # request already aborted

        req_id = "aborted-req"
        shape = (1, 4, 8)
        dtype = torch.float16
        t = torch.zeros(shape, dtype=dtype)
        shm = write_embeddings(t, req_id)
        try:
            runner.store_mm_embeddings([(req_id, shape, dtype)])
        finally:
            cleanup_embeddings(shm)
            _cleanup_if_exists(req_id)

        assert req_id not in runner.pending_mm_embeddings

    def test_stores_embedding_for_active_request(self):
        """Embedding for an active request must be written to pending_mm_embeddings."""
        from unittest.mock import MagicMock

        runner = self._make_runner()
        runner.requests = {"active-req": MagicMock()}

        req_id = "active-req"
        shape = (1, 4, 8)
        dtype = torch.float16
        t = torch.ones(shape, dtype=dtype)
        shm = write_embeddings(t, req_id)
        try:
            runner.store_mm_embeddings([(req_id, shape, dtype)])
        finally:
            cleanup_embeddings(shm)
            _cleanup_if_exists(req_id)

        assert req_id in runner.pending_mm_embeddings
        assert runner.pending_mm_embeddings[req_id].shape == torch.Size(shape)
