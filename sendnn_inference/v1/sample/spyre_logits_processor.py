import itertools
from dataclasses import dataclass, field
from typing import Sequence, Union

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (
    BUILTIN_LOGITS_PROCESSORS,
    STR_POOLING_REJECTS_LOGITSPROCS,
    BatchUpdate,
    LogitsProcessor,
    _load_custom_logitsprocs,
)
from vllm.v1.sample.logits_processor.state import BatchUpdateBuilder, LogitsProcessors

logger = init_logger(__name__)


@dataclass(frozen=True)
class SpyreBatchUpdate(BatchUpdate):
    """Extends BatchUpdate with pause/resume lifecycle events."""

    # (dense_index, req_id) pairs — request was temporarily removed from the
    # active batch; its logitproc state should be saved and not destroyed.
    paused: list[tuple[int, str]] = field(default_factory=list)
    # (dense_index, req_id) pairs — request is returning to the active batch;
    # its previously saved logitproc state should be restored at dense_index.
    resumed: list[tuple[int, str]] = field(default_factory=list)
    # req_ids for requests that finished while paused and whose saved state
    # should be discarded without being restored to an active slot.
    finished_paused: list[str] = field(default_factory=list)


class SpyreBatchUpdateBuilder(BatchUpdateBuilder):
    """Extends BatchUpdateBuilder with pause/resume tracking."""

    def __init__(self) -> None:
        super().__init__()
        self._paused: list[tuple[int, str]] = []
        self._resumed: list[tuple[int, str]] = []
        self._finished_paused: list[str] = []

    def pause_append(self, dense_index: int, req_id: str) -> None:
        self._paused.append((dense_index, req_id))

    def resume_append(self, dense_index: int, req_id: str) -> None:
        self._resumed.append((dense_index, req_id))

    def finished_paused_append(self, req_id: str) -> None:
        self._finished_paused.append(req_id)

    def get_and_reset(self, batch_size: int) -> SpyreBatchUpdate | None:
        paused, self._paused = self._paused, []
        resumed, self._resumed = self._resumed, []
        finished_paused, self._finished_paused = self._finished_paused, []
        base = super().get_and_reset(batch_size)
        if base is None and not paused and not resumed and not finished_paused:
            return None
        if base is None:
            return SpyreBatchUpdate(
                batch_size=batch_size,
                removed=[],
                added=[],
                moved=[],
                paused=paused,
                resumed=resumed,
                finished_paused=finished_paused,
            )
        return SpyreBatchUpdate(
            batch_size=base.batch_size,
            removed=base.removed,
            added=base.added,
            moved=base.moved,
            paused=paused,
            resumed=resumed,
            finished_paused=finished_paused,
        )


def build_logitsprocs_for_cb(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    batch_size: int,
    custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] | None = None,
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug(
            "Skipping logits processor loading because pooling models"
            " do not support logits processors."
        )
        return LogitsProcessors()
    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)

    return LogitsProcessors(
        LogitProcessorWrapper(logit_processor, vllm_config, device, is_pin_memory, batch_size)
        for logit_processor in itertools.chain(
            BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes
        )
    )


class LogitProcessorWrapper(LogitsProcessor):
    """Per-request logits processor manager for the persistent CB batch.

    Maintains one inner LogitsProcessor instance per active dense slot.
    Pause/resume events save and restore exact per-request state so that
    temporarily held-back requests do not lose their generation history.
    """

    def __init__(
        self,
        logit_processor: type[LogitsProcessor],
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
        batch_size: int,
    ):
        self._factory = lambda: logit_processor(vllm_config, device, is_pin_memory)
        self.logitprocs: list[LogitsProcessor] = [self._factory() for _ in range(batch_size)]
        # Saved logitproc objects for paused requests, keyed by req_id.
        self._saved: dict[str, LogitsProcessor] = {}
        self._is_argmax_invariant: bool = self.logitprocs[0].is_argmax_invariant()
        self._prefill_index: int | None = None

    def is_argmax_invariant(self) -> bool:
        return self._is_argmax_invariant

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        # Some LogitsProcessors (e.g. MinTokensLogitsProcessor) require
        # update_state to be called even when batch_update is None.
        update_called = {i: False for i in range(len(self.logitprocs))}

        if batch_update is not None:
            for index, params, prompt_tok_ids, out_tok_ids in batch_update.added:
                update_called[index] = True
                if self.logitprocs[index] is None:
                    self.logitprocs[index] = self._factory()
                self.logitprocs[index].update_state(
                    BatchUpdate(
                        batch_size=1,
                        removed=[],
                        moved=[],
                        added=[(0, params, prompt_tok_ids, out_tok_ids)],
                    )
                )

            for index in batch_update.removed:
                update_called[index] = True
                self.logitprocs[index].update_state(
                    BatchUpdate(batch_size=1, removed=[0], moved=[], added=[])
                )

            for index, req_id in getattr(batch_update, "resumed", []):
                assert req_id in self._saved
                self.logitprocs[index] = self._saved.pop(req_id)

            for index, req_id in getattr(batch_update, "paused", []):
                self._saved[req_id] = self.logitprocs[index]
                self.logitprocs[index] = None

            for req_id in getattr(batch_update, "finished_paused", []):
                assert req_id in self._saved
                self._saved.pop(req_id)

            for adx, bdx, _ in batch_update.moved:
                update_called[adx], update_called[bdx] = update_called[bdx], update_called[adx]
                self.logitprocs[adx], self.logitprocs[bdx] = (
                    self.logitprocs[bdx],
                    self.logitprocs[adx],
                )

        for index, called in update_called.items():
            if not called and self.logitprocs[index] is not None:
                self.logitprocs[index].update_state(None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._prefill_index is not None:
            logits = self.logitprocs[self._prefill_index].apply(logits)
            self._prefill_index = None
            return logits

        for i in range(logits.shape[0]):
            logits[i] = self.logitprocs[i].apply(logits[i].unsqueeze(0))
        return logits

    def set_prefill_index(self, idx: int) -> None:
        self._prefill_index = idx
