"""Contains utilities for LLM caching"""

import os
import time
from typing import NamedTuple

from spyre_util import EmbeddingWarmupShapes, ModelInfo
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory


def force_engine_shutdown(llm: LLM):
    force_engine_core_shutdown(llm.llm_engine.engine_core)


def force_engine_core_shutdown(engine_core):
    """
    🌶️🌶️🌶️
    This hack is here because of an issue in vllm 0.9.2+ where a circular
    reference occurs in vllm.executor.ray_utils if ray is not installed. This
    circular reference holds a copy of the vllm config which contains a
    reference to the LLM, which means it can never be garbage collected.
    Since vllm.LLM relies on garbage collection to shut down its engine, the
    engine never shuts down. When running tensor parallel workloads, if the
    engine is never shut down then the TP worker processes are never killed.
    When the TP worker processes are held open, all future attempts to create a
    new engine will fail with an EADDRINUSE error.
    🌶️🌶️🌶️
    """
    try:
        engine_core.shutdown()
    finally:
        # Cached-engine transitions happen outside the per-test cleanup fixture.
        # Tear down distributed state immediately so the next engine/server
        # setup does not inherit stale resources from the previous run.
        cleanup_dist_env_and_memory()

        # Spyre device teardown is not always instantaneous. Give the runtime
        # a short grace period before the next cached config starts.
        if os.environ.get("SENDNN_INFERENCE_DYNAMO_BACKEND") == "sendnn":
            time.sleep(2)


def sort_tests_for_llm_caching(items: list) -> None:
    """Sorts a list of pytest cases based on the LLM parameterizations.

    This allows us to group tests together that use the same model and config,
    which means they can reuse the underlying LLM. Then we can cache the LLM
    across tests to save time.

    This is important because spinning up a new vLLM engine from scratch takes
    a decent amount of time, even with the torch compilation cache active. LLM
    creation dominates the runtime of our test suites.

    This sorts the `items` list in-place.
    """
    items.sort(key=SortKey.from_item)


class SortKey(NamedTuple):
    """Sort key that groups by runtime configuration.

    The order of attributes is important here and controls the test
    grouping.
    """

    cache_priority: int
    cache_type: str  # None (empty str), online, llm, engine
    backend: str = ""
    model: str = ""
    tp_size: int = 1
    use_cp: bool = False
    use_pc: bool = False
    max_model_len: int = 0
    max_num_seqs: int = 0
    num_blocks: int = 0
    max_num_batched_tokens: int = 0
    structured_output_backend: str = ""
    warmup_shapes: EmbeddingWarmupShapes | None = None

    @staticmethod
    def from_item(item) -> "SortKey":
        cache_type = SortKey._get_cache_type(item)
        if not cache_type:
            # Don't add any extra re-ordering logic for tests that won't utilize
            # the cache
            return SortKey(cache_priority=0, cache_type=cache_type)

        if not hasattr(item, "callspec"):
            # This isn't great- we probably want to cache but can't because the
            # test has no parameters at all
            return SortKey(cache_priority=0, cache_type="")

        use_pc = SortKey._uses_pc(item)
        warmup_shapes = SortKey._get_warmup_shapes(item)

        if warmup_shapes[0][0] == -1:
            sort_kwargs = {
                "max_model_len": SortKey._get_max_model_len(item),
                "max_num_seqs": SortKey._get_max_num_seqs(item),
            }
        else:
            sort_kwargs = {
                "warmup_shapes": warmup_shapes,
            }

        return SortKey(
            cache_priority=SortKey._get_cache_priority(cache_type),
            cache_type=cache_type,
            model=SortKey._get_model(item),
            backend=SortKey._get_backend(item),
            tp_size=SortKey._get_tp_size(item),
            use_pc=use_pc,
            num_blocks=SortKey._get_num_blocks(item),
            max_num_batched_tokens=SortKey._get_max_num_batched_tokens(item),
            structured_output_backend=SortKey._get_structured_output_backend(item),
            **sort_kwargs,
        )

    @staticmethod
    def _get_cache_type(item) -> str:
        # If not an e2e test then assume no cache
        if "e2e" not in item.listnames():
            return ""

        if "remote_openai_server" in item.fixturenames:
            # (Not actually caching these yet, but can in future)
            return "online"

        if "use_llm_cache" in item.fixturenames:
            return "llm"

        # All of the *_steps.py tests use cached engines to test scheduling
        # logic
        filename = [i for i in item.listnames() if i.endswith(".py")][0]
        if "steps.py" in filename:
            return "engine"

        # Else shouldn't be using any cache
        return ""

    @staticmethod
    def _get_cache_priority(cache_type: str) -> int:
        # Sort online tests before cached LLM tests so the server-backed path
        # does not have to follow a same-process SenDNN engine teardown.
        cache_order = {
            "": 0,
            "online": 1,
            "llm": 2,
            "engine": 3,
        }
        return cache_order[cache_type]

    @staticmethod
    def _uses_pc(item) -> bool:
        """True if the test uses prefix caching.
        Checks for the pytest.mark.prefix_caching mark."""
        markers = {mark.name for mark in item.own_markers}
        return "prefix_caching" in markers or "pc" in markers

    def _get_max_num_batched_tokens(item) -> int:
        """Chunk size for chunked prefill, if enabled"""
        params = item.callspec.params
        if "max_num_batched_tokens" in params:
            SortKey._assert_param(
                isinstance(params["max_num_batched_tokens"], int),
                "max_num_batched_tokens must be an int",
                item,
            )
            return params["max_num_batched_tokens"]
        return 0

    @staticmethod
    def _get_max_model_len(item) -> int:
        params = item.callspec.params
        if "max_model_len" in params:
            SortKey._assert_param(
                isinstance(params["max_model_len"], int),
                "max_model_len must be an int",
                item,
            )
            return params["max_model_len"]
        # Put `-1` to indicate that this couldn't be found
        return -1

    @staticmethod
    def _get_max_num_seqs(item) -> int:
        params = item.callspec.params
        if "max_num_seqs" in params:
            SortKey._assert_param(
                isinstance(params["max_num_seqs"], int),
                "max_num_seqs must be an int",
                item,
            )
            return params["max_num_seqs"]
        # Put `-1` to indicate that this couldn't be found
        return -1

    @staticmethod
    def _get_warmup_shapes(item) -> list[tuple[int, int, int]]:
        key = "warmup_shapes"
        params = item.callspec.params
        if key in params:
            shapes = params[key]
            SortKey._assert_param(
                isinstance(shapes, list), "Warmup shape must be a list of tuples", item
            )
            SortKey._assert_param(
                isinstance(shapes[0], tuple),
                "Warmup shape must be a list of tuples",
                item,
            )
            return params[key]
        # Use -1s to indicate that this couldn't be found
        return [
            (-1, -1, -1),
        ]

    @staticmethod
    def _get_tp_size(item) -> int:
        TP_KEYS = ["tp_size", "tensor_parallel_size", "tp"]
        params = item.callspec.params
        for key in TP_KEYS:
            if key in params:
                SortKey._assert_param(isinstance(params[key], int), "tp size must be an int", item)
                return params[key]
        # Assume no TP if not set
        return 1

    @staticmethod
    def _get_model(item) -> str:
        MODEL_KEYS = ["model", "model_name"]
        params = item.callspec.params
        for key in MODEL_KEYS:
            if key in params:
                SortKey._assert_param(
                    isinstance(params[key], str | ModelInfo),
                    "model must be a string or ModelInfo",
                    item,
                )
                model_or_info = params[key]
                if isinstance(model_or_info, ModelInfo):
                    return model_or_info.name
                return model_or_info
        # No assumption about default model, we likely don't need an llm if this
        # isn't set
        return ""

    @staticmethod
    def _get_backend(item) -> str:
        if "backend" in item.callspec.params:
            backend = item.callspec.params["backend"]
            # if isinstance(backend, tuple) and len(backend) == 1:
            #     backend = backend[0]

            SortKey._assert_param(isinstance(backend, str), "backend must be a string.", item)
            return backend
        # If backend isn't given then this is likely a spyre-only test
        return "sendnn"

    @staticmethod
    def _get_num_blocks(item) -> int:
        if "available_blocks" in item.callspec.params:
            blocks = item.callspec.params["available_blocks"]
            SortKey._assert_param(
                isinstance(blocks, int | None), "available_blocks must be an optional int.", item
            )
            return blocks if blocks is not None else 0
        # Most tests don't use this param
        return 0

    @staticmethod
    def _get_structured_output_backend(item) -> str:
        """Extract structured output backend from test parameters."""
        if "structured_output_backend" in item.callspec.params:
            backend = item.callspec.params["structured_output_backend"]
            SortKey._assert_param(
                isinstance(backend, str),
                "structured_output_backend must be a string.",
                item,
            )
            return backend
        # Most tests don't use structured outputs
        return ""

    @staticmethod
    def _assert_param(condition, message, item):
        assert condition, (
            message + f"\n\n\tTest: {item.listnames()}\n\n\tParams: {item.callspec.params}"
        )
