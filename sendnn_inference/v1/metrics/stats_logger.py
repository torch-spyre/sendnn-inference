import dataclasses
import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import async_llm, llm_engine
from vllm.v1.metrics.loggers import StatLoggerBase, StatLoggerManager
from vllm.v1.metrics.stats import (
    FinishedRequestStats,
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
)

from sendnn_inference import envs as envs_spyre

logger = init_logger(__name__)


@dataclasses.dataclass
class PerfRecord:
    """A record for request_metrics.jsonl.
    Contains info about a single finished request"""

    # ISO timestamp w/ milliseconds
    timestamp: str
    # timing info
    engine_stats: FinishedRequestStats
    # time spent pre-emptied for other prefills
    prefill_interrupt_seconds: float
    # ITL calculated without the prefill interrupts
    decode_only_itl_seconds: float

    # key names to append with a time unit during json serialization
    _TIME_KEYS = [
        "e2e_latency",
        "queued_time",
        "prefill_time",
        "inference_time",
        "decode_time",
        "mean_time_per_output_token",
    ]

    def to_json(self) -> str:
        json_dict = dataclasses.asdict(self)

        # Flatten the engine stats into the top level
        engine_dict = json_dict.pop("engine_stats")
        json_dict.update(engine_dict)

        # add _seconds onto the timing info from the engine
        for k in self._TIME_KEYS:
            if k in json_dict:
                json_dict[k + "_seconds"] = json_dict.pop(k)

        return json.dumps(json_dict)


class FileStatLogger(StatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_index=0):
        self.enabled = envs_spyre.SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED

        perf_dir = Path(envs_spyre.SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR)
        if not perf_dir.exists():
            perf_dir.mkdir(parents=True)

        self.perf_file = (
            Path(envs_spyre.SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR) / "request_metrics.jsonl"
        )

        if self.enabled and engine_index == 0:
            logger.info(
                "Initializing sendnn-inference perf debug logger. Writing perf info to: %s",
                str(self.perf_file),
            )

        # Clear any old metrics out first
        if self.perf_file.exists():
            self.perf_file.unlink()
        self.perf_file.touch()

        self.iso_format = "%Y-%m-%dT%H:%M:%S.%f"

        self._prefill_tuples: list[tuple[float, float]] = []
        self._max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self._last_ts: float = 0

        self.open_file_pointer = self.perf_file.open("a")

    def __del__(self):
        self.open_file_pointer.close()

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        if not self.enabled or engine_idx != 0:
            # Only log from rank 0
            return

        if iteration_stats is None:
            return

        if iteration_stats.num_prompt_tokens > 0:
            self._save_prefill_time(iteration_stats)
        self._last_ts = iteration_stats.iteration_timestamp

        if not iteration_stats.finished_requests:
            # Only log finished requests
            return

        # Convert float timestamp to human readable string
        text_timestamp = datetime.fromtimestamp(iteration_stats.iteration_timestamp).strftime(
            self.iso_format
        )[:-3]

        records_to_write: list[str] = []
        for r in iteration_stats.finished_requests:
            # Calculate some estimates to add to the engine stats
            estimated_prefill_interrupt = self.estimate_prefill_interrupt_lower_bound(r)

            estimated_decode_itl = (r.decode_time - estimated_prefill_interrupt) / max(
                r.num_generation_tokens - 1, 1
            )

            record = PerfRecord(
                timestamp=text_timestamp,
                engine_stats=r,
                decode_only_itl_seconds=estimated_decode_itl,
                prefill_interrupt_seconds=estimated_prefill_interrupt,
            )
            records_to_write.append(record.to_json())

        self.open_file_pointer.write("\n".join(records_to_write) + "\n")
        self.open_file_pointer.flush()

    def log_engine_initialized(self):
        pass

    def _save_prefill_time(self, iteration_stats: IterationStats):
        """If this iteration was a prefill, then save the a tuple of the current
        time and prefill time. This will be used later to estimate a lower bound
        of the amount of time that other sequences were
        interrupted for this prefill to happen.

        This is only relevant because the batching implementation has to pause
        the running batch of decoding sequences to prefill a single sequence.
        """
        maybe_prefill_time = iteration_stats.iteration_timestamp - self._last_ts
        # TTFT here includes queueing and we don't have access to the iteration
        # duration itself so we have to try to calculate our own prefill time.
        # If we calculate an interval that was less than the reported TTFT, then
        # use it as the prefill time
        maybe_prefill_time = min(maybe_prefill_time, iteration_stats.time_to_first_tokens_iter[0])

        # Tuple is (timestamp, prefill_time)
        self._prefill_tuples.append((iteration_stats.iteration_timestamp, maybe_prefill_time))
        if len(self._prefill_tuples) > 2 * self._max_batch_size:
            # Delete older prefills, we can't hold everything in memory
            # Not guaranteed to be lossless
            self._prefill_tuples.pop(0)

    def estimate_prefill_interrupt_lower_bound(
        self, finished_request: FinishedRequestStats
    ) -> float:
        """Returns a lower bound estimate on the time (in ms) that this request
        was interrupted for other requests to prefill to join the batch"""
        estimated_prefill_interrupt: float = 0

        # NB: use current time instead of iteration timestamp to ensure that we
        # exclude current request's prefill
        slop = 0.001
        decode_start_time = time.time() - finished_request.decode_time + slop
        for i in range(len(self._prefill_tuples)):
            if self._prefill_tuples[i][0] > decode_start_time:
                # Sum up all prefills past decode start time
                estimated_prefill_interrupt = sum(r[1] for r in self._prefill_tuples[i:])
                break
        return estimated_prefill_interrupt


def file_stat_logger_factory(config: VllmConfig, engine_index=0) -> FileStatLogger:
    """Factory method accepted by vllm engine initializers"""
    return FileStatLogger(config, engine_index)


def patch_async_llm_stat_loggers():
    """
    🌶️🌶️🌶️
    Platforms cannot alter the initialization of a vllm engine, and the
    `stat_loggers` parameter is not user-settable via `EngineArgs`.

    So we resort to patching the initialization of the StatsLoggerManager to
    inject our own stats logger. This _should_ also be compatible with versions
    of vllm prior to the addition of `stats_loggers` engine parameter.
    🌶️🌶️🌶️
    """
    logger.debug("Setting up perf logger injection")
    original_init = StatLoggerManager.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        logger.debug("Injecting sendnn-inference perf logger factory")
        if "custom_stat_loggers" not in kwargs or kwargs["custom_stat_loggers"] is None:
            kwargs["custom_stat_loggers"] = []

        kwargs["custom_stat_loggers"].append(file_stat_logger_factory)

        original_init(self, *args, **kwargs)

    async_llm.StatLoggerManager.__init__ = new_init  # ty: ignore[invalid-assignment]
    llm_engine.StatLoggerManager.__init__ = new_init  # ty: ignore[invalid-assignment]
