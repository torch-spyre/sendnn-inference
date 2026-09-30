"""Per-rank, cross-request cache of vision encoder outputs.

The (expensive) vision tower + projector turn an image into packed feature vectors
(shape ``[num_image_tokens, emb_dim]``). Those features depend only on the image,
so they are cached here keyed by the multimodal content hash
(``MultiModalFeatureSpec.identifier``, a.k.a. mm_hash).

On a later request containing the same image, the caller reuses the cached features
and merges them into freshly-computed text embeddings, skipping the vision tower.
See ``spyre_model_runner._compute_and_cache_mm_embeddings``.

The cache is a byte-bounded LRU. It lives on each TP rank independently; because
every rank processes the identical request stream and stores identically-sized
tensors, the ranks' caches stay in lock-step (same contents, same evictions) with
no cross-rank coordination. Cached tensors are kept on CPU and cloned on insert so
they are detached from any request-scoped buffer.
"""

from collections import OrderedDict
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Identifiers used for warmup features must never be cached: they are dummy
# images and would poison real lookups (and are reused across models).
_WARMUP_IDENTIFIER_PREFIX = "MM-warmup"


def cacheable_identifiers(mm_features: Any) -> list[str]:
    """Return the mm_hash ``identifier`` of each cacheable image in the request.

    The identifier (mm_hash) alone keys the cache. Features without an
    ``mm_position`` (not a real image placeholder) and warmup/non-cacheable
    identifiers are skipped (see ``MMEncoderCache.is_cacheable``).
    """
    identifiers: list[str] = []
    for feat in mm_features or []:
        identifier = getattr(feat, "identifier", None)
        if getattr(feat, "mm_position", None) is None or not MMEncoderCache.is_cacheable(
            identifier
        ):
            continue
        identifiers.append(identifier)  # ty: ignore[invalid-argument-type]
    return identifiers


class MMEncoderCache:
    """Byte-bounded LRU mapping mm_hash -> packed image features (CPU)."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = max(0, capacity_bytes)
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._nbytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.capacity_bytes > 0

    @staticmethod
    def is_cacheable(identifier: str | None) -> bool:
        return bool(identifier) and not identifier.startswith(_WARMUP_IDENTIFIER_PREFIX)

    def get(self, identifier: str) -> torch.Tensor | None:
        """Return the cached features for *identifier*, marking most-recently-used.

        Does not update hit/miss counters — call :meth:`record_lookup` once per
        request after deciding hit vs. miss.
        """
        tensor = self._store.get(identifier)
        if tensor is not None:
            self._store.move_to_end(identifier)
        return tensor

    def put(self, identifier: str, tensor: torch.Tensor) -> None:
        if not self.enabled or not self.is_cacheable(identifier):
            return
        tensor = tensor.detach().to("cpu").contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        # A single entry larger than the whole budget is simply not cached.
        over = nbytes > self.capacity_bytes
        logger.debug(
            "MM encoder cache: entry '%s' size=%.2f MiB, budget=%.2f MiB "
            "(SENDNN_INFERENCE_MM_ENCODER_CACHE_MB) — %s",
            identifier,
            nbytes / 1024 / 1024,
            self.capacity_bytes / 1024 / 1024,
            "OVER budget → NOT cached" if over else "under budget → cached",
        )
        if over:
            return
        if identifier in self._store:
            self._nbytes -= self._store[identifier].numel() * self._store[identifier].element_size()
            self._store.pop(identifier)
        self._store[identifier] = tensor
        self._nbytes += nbytes
        self._evict_to_fit()

    def _evict_to_fit(self) -> None:
        while self._nbytes > self.capacity_bytes and self._store:
            _, evicted = self._store.popitem(last=False)
            self._nbytes -= evicted.numel() * evicted.element_size()

    def record_lookup(self, hit: bool) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._store
