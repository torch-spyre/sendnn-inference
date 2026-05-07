import sys
from string import Template
import multiprocessing
import importlib.metadata

# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get("triton"):
        del sys.modules["triton"]

import argparse
import math
import operator
import os
from typing import TYPE_CHECKING, cast, Literal

import torch
import huggingface_hub
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from sendnn_inference.argparse_utils import ConditionalDefaultManager

if TYPE_CHECKING:
    # NB: We can't eagerly import many things from vllm since vllm.config
    # will import this file. These would lead to circular imports
    from vllm.config import ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.inputs import EngineInput, TokensInput
else:
    ModelConfig = None
    VllmConfig = None
    SamplingParams = None
    PoolingParams = None
    EngineInput = None
    TokensInput = None
from vllm.platforms import Platform, PlatformEnum

import sendnn_inference.envs as envs_spyre
from sendnn_inference.compilation_utils import handle_disable_compilation

logger = init_logger(__name__)

THREADING_ENVS = [
    "OMP_NUM_THREADS",
    # "TORCHINDUCTOR_COMPILE_THREADS", # vLLM wants this set to 1
    "DT_PARALLEL_THREADS",  # affects the compilation during warmup
    # set these for good measure
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
]


# Needed by vllm/model_executor/layers/pooler.py:562
# Copied from vllm/utils/__init__.py
class _StreamPlaceholder:
    def __init__(self):
        self.synchronize = lambda: None


class SpyrePlatform(Platform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"
    # compressed-tensors supported by
    # https://github.com/foundation-model-stack/fms-model-optimizer/blob/main/fms_mo/aiu_addons/__init__.py
    supported_quantization: list[str] = ["gptq", "compressed-tensors"]
    _warmup_shapes: tuple[dict[str, int], ...] | None = None
    _block_size: Literal[64] = 64  # hardcoded Spyre constraint for now
    # TODO: this `None` is dangerous
    _config: VllmConfig = None  # ty: ignore[invalid-assignment]
    _torch_sendnn_configured: bool = False

    _max_batch_tkv_limit: int = 0

    # Backend for dynamic compilation ops
    # See vllm batched_count_greater_than method
    simple_compile_backend: str = envs_spyre.SENDNN_INFERENCE_SIMPLE_COMPILE_BACKEND

    # Needed by vllm/model_executor/layers/pooler.py:562
    current_stream = lambda _: _StreamPlaceholder()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "spyre"

    @classmethod
    def log_server_boot(cls, vllm_config: VllmConfig) -> None:
        # Only log in main process (not in TP workers)
        if multiprocessing.current_process().name != "MainProcess":
            return

        # yapf: disable
        logo_template = Template(
            template="\n       ${w}█     █     █▄   ▄█${r}       ${red}▄█▀▀█▄${r}  ${orange}█▀▀▀█▄${r}  ${yellow}█   █${r}  ${green}█▀▀▀█▄${r}  ${blue}█▀▀▀▀${r}\n" # noqa: E501
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}       ${red}▀▀▄▄▄${r}   ${orange}█▄▄▄█▀${r}  ${yellow}▀▄ ▄▀${r}  ${green}█▄▄▄█▀${r}  ${blue}█▄▄▄${r}   version ${w}%s${r}\n" # noqa: E501
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}            ${red}█${r}  ${orange}█${r}        ${yellow}▀█▀${r}   ${green}█ ▀█▄${r}   ${blue}█${r}      model   ${w}%s${r}\n" # noqa: E501
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}       ${red}▀▄▄▄█▀${r}  ${orange}█${r}         ${yellow}█${r}    ${green}█   ▀█${r}  ${blue}█▄▄▄▄${r}\n" # noqa: E501
        )
        # yapf: enable
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
            "red": "\033[91m",  # red (rainbow start)
            "orange": "\033[38;5;208m",  # orange
            "yellow": "\033[93m",  # yellow
            "green": "\033[92m",  # green
            "blue": "\033[94m",  # blue (rainbow end)
        }

        message = logo_template.substitute(colors)

        version = importlib.metadata.version("sendnn_inference")

        model_name = vllm_config.model_config.model if vllm_config.model_config else "N/A"

        logger.info(message, version, model_name)

    @classmethod
    def import_kernels(cls) -> None:
        # Workaround torch.accelerator.empty_cache for torch 2.7.1 and vllm v0.18.0 compatibility
        setattr(torch.accelerator, "empty_cache", lambda: None)  # noqa

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @classmethod
    def get_max_batch_tkv_limit(cls) -> int:
        if cls._max_batch_tkv_limit == 0:
            # For spawned subprocesses, we need to grab the TKV limit from the environment
            cls._set_batch_tkv_limit_from_env()
        return cls._max_batch_tkv_limit

    @classmethod
    def get_total_spyre_blocks(cls, vllm_config: VllmConfig) -> int:
        """Returns the total number of KV cache blocks available for spyre.
        This currently returns the number of blocks required for a full-sized
        batch, which may be greater than the available memory.

        Until a correct available memory api is available, the number of blocks
        must be overridden with a known good value via
        cache_config.num_gpu_blocks_override
        """
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        block_size = SpyrePlatform.get_block_size()
        max_blocks_per_seq = max_model_len // block_size
        num_blocks_full_batch = max_batch_size * max_blocks_per_seq

        blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        if blocks_override is not None and blocks_override > 0:
            num_blocks = blocks_override
            # Total number of blocks needs to be a multiple of the batch size on Spyre
            # -> round down to not exceed the Spyre cards hard-coded limits of detected models
            old_num_blocks = num_blocks
            num_blocks = math.floor(num_blocks / max_batch_size) * max_batch_size
        else:
            # Note on "+1": We need to add one additional block used exclusively for padding (idx 0)
            num_blocks = num_blocks_full_batch + 1
            # Total number of blocks needs to be a multiple of the batch size on Spyre
            # -> round up as not among detected models (rounding down might cut the padding block)
            old_num_blocks = num_blocks
            num_blocks = math.ceil(num_blocks / max_batch_size) * max_batch_size

        if num_blocks != old_num_blocks:
            logger.info(
                "Spyre constraint: num blocks rounded from %d to %d (multiple of batch size=%d)",
                old_num_blocks,
                num_blocks,
                max_batch_size,
            )

        # As we drop the block reservation for chunked prefill the number of available blocks
        # needs to be at least as big as the smaller of the batch tkv limit
        # (VLLM_DT_MAX_BATCH_TKV_LIMIT) and a full batch (max_num_seqs * max_model_len)
        num_blocks_batch_tkv_limit = cls.get_max_batch_tkv_limit() // block_size
        # Note on "+1": We need to add one additional block used exclusively for padding (idx 0)
        min_req_num_blocks = min(num_blocks_full_batch, num_blocks_batch_tkv_limit) + 1

        # min_req_num_blocks := minimum required number of blocks
        if num_blocks < min_req_num_blocks:
            raise ValueError(
                f"Number of pages available on Spyre {num_blocks} is not "
                f"enough to serve the current model (need at least "
                f"{min_req_num_blocks} pages)."
            )

        max_concurrency = num_blocks * block_size / max_model_len
        backend = "Spyre" if envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND == "sendnn" else "CPU"
        logger.info("%s KV cache size: %s tokens", backend, num_blocks * block_size)
        logger.info(
            "Maximum concurrency for %s tokens per request: %.2fx",
            str(max_model_len),
            max_concurrency,
        )

        return num_blocks

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # 🌶️🌶️🌶️ Patch in our perf logger before the engine is created
        from sendnn_inference.v1.metrics import patch_async_llm_stat_loggers

        patch_async_llm_stat_loggers()

        # In case vllm passes a default vllm_config to us.
        # This happens when get_current_vllm_config is called
        # without setting the vllm config through
        # set_current_vllm_config
        if vllm_config.model_config is None:
            return
        
        # print startup logo
        cls.log_server_boot(vllm_config)

        cls._config = vllm_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        is_decoder = model_config.runner_type == "generate"

        is_pooling = model_config.runner_type == "pooling"

        if not is_decoder and not is_pooling:
            raise ValueError("Only the 'generate' and 'pooling' runners are supported")

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "sendnn_inference.v1.worker.spyre_worker.SpyreWorker"

        cls._check_threading_config(parallel_config.world_size)

        # set env vars based on the model
        if is_decoder:
            os.environ["FLEX_OVERWRITE_NMB_FRAME"] = "true"
            os.environ["COMPILATION_MODE"] = "offline_decoder"
        if is_pooling:
            os.environ["FLEX_OVERWRITE_NMB_FRAME"] = "false"
            os.environ["COMPILATION_MODE"] = "offline"

        logger.info("Using backend: %s", envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND)
        if envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND == "sendnn_compile_only":
            os.environ["FLEX_DEVICE"] = "COMPILE"

        if is_decoder:
            scheduler_config.scheduler_cls = (
                "sendnn_inference.v1.core.scheduler.ChunkedPrefillSpyreScheduler"
            )

            if (
                vllm_config.model_config.quantization
                and vllm_config.scheduler_config.max_num_seqs == 1
            ):
                raise ValueError("Batch size 1 not supported for fp8 continuous batching.")
        else:
            # Static batching or embedding model.
            # Override --max-num-seqs to the biggest warmup batch size
            # And override --max-model-len to the biggest warmup sequence
            cls._warmup_shapes = None
            spyre_warmup_shapes = cls.get_warmup_shapes(scheduler_config)
            max_batch_size = 0
            max_seq_len = 0
            for shape in spyre_warmup_shapes:
                max_batch_size = max(max_batch_size, shape["batch_size"])
                max_seq_len = max(max_seq_len, shape["prompt_length"])

            # verify that warmup shapes are not too large
            model_config.get_and_verify_max_len(max_model_len=max_seq_len)

            # override stuff
            model_config.max_model_len = max_seq_len
            scheduler_config.max_num_seqs = max_batch_size
            # unsetting this config as it was only set to pass vllm scheduler's max_model_len check
            vllm_config.scheduler_config.enable_chunked_prefill = False

            scheduler_config.scheduler_cls = (
                "sendnn_inference.v1.core.scheduler.PoolingSpyreScheduler"
            )

        # Apply model-specific configurations using the registry
        # Only when running on Spyre device (sendnn backend)
        if cls.is_backend_sendnn_enabled():
            from sendnn_inference.config.model_registry import get_model_registry

            registry = get_model_registry()

            # For static batching (pooling models), pass warmup shapes for validation
            warmup_shape_tuples = (
                [(ws["prompt_length"], ws["batch_size"]) for ws in cls._warmup_shapes]
                if cls._warmup_shapes
                else None
            )
            configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shape_tuples)

            if configurator:
                config_summary = configurator.configure(vllm_config)
                logger.info(config_summary.format_log_message())
            else:
                error_msg = f"No model-specific configuration found for '{model_config.model}'"
                if envs_spyre.SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG:
                    raise RuntimeError(
                        f"{error_msg}. SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG is set, "
                        "which requires a known configuration to be found."
                    )
                logger.debug(error_msg)

        else:
            logger.debug(
                "Model registry validation skipped for backend '%s'. "
                "Registry validation is only performed for 'sendnn'.",
                envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND,
            )

        # TODO: try to support async scheduling
        scheduler_config.async_scheduling = False

        # To disable any paged attention ops in the base scheduler, we:
        # - Set the block size (in tokens) to the maximum sequence length
        #       so that the scheduler thinks an entire sequence will fit in
        #       one single block.
        # - For pooling models, set `max_num_batched_tokens` to the size of a
        #       full batch of full length requests, so that the scheduler will
        #       always have token budget available to schedule a full batch
        # - For generative models, set `max_num_batched_tokens` to the chunk
        #       chunk size used for chunked prefill.
        if cache_config is not None:
            if not is_decoder:
                scheduler_config.max_num_batched_tokens = (
                    model_config.max_model_len * scheduler_config.max_num_seqs
                )
                cache_config.block_size = model_config.max_model_len  # ty: ignore[invalid-assignment]
                vllm_config.cache_config.enable_prefix_caching = False

            else:
                cache_config.block_size = cls._block_size
                # Set VLLM_DT_CHUNK_LEN based on scheduler_config.max_num_batched_tokens
                os.environ["VLLM_DT_CHUNK_LEN"] = str(scheduler_config.max_num_batched_tokens)

                assert scheduler_config.max_num_batched_tokens % cls._block_size == 0, (
                    "`max_num_batched_tokens` must"
                    f" be divisible by the block size ({cls._block_size}) "
                    "to enable chunked prefill. It was set to "
                    f"`{scheduler_config.max_num_batched_tokens}`. Please "
                    "set `--max-num-batched-tokens` to a number that satisfies "
                    "this constraint."
                )
                if cache_config.num_gpu_blocks_override is None:
                    cache_config.num_gpu_blocks_override = cls.get_total_spyre_blocks(vllm_config)
            cache_config.user_specified_block_size = True

        logger.info(
            "Configurations for Spyre. max_model_len=%d, max_num_seqs=%d, block_size=%d, "
            "max_num_batched_tokens=%d, enable_chunked_prefill=%r, enable_prefix_caching=%r",
            model_config.max_model_len,
            scheduler_config.max_num_seqs,
            cache_config.block_size,
            scheduler_config.max_num_batched_tokens,
            is_decoder,
            cache_config.enable_prefix_caching,
        )

        # set env vars for torch_sendnn to consume
        os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(vllm_config.model_config.max_model_len)
        if vllm_config.model_config.max_model_len > 32 * 1024:
            logger.warning(
                "Max context length is too big. Currently only 32K (32768) context length is "
                "supported on Spyre for continuous batching. Results might be off!"
            )
        # min value 2 needed for VLLM_DT_MAX_BATCH_SIZE (compiler constraint)
        # Note that we can still have decodes of batch size 1 as the env var
        # only concerns the max batch size.
        os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(
            max(vllm_config.scheduler_config.max_num_seqs, 2)
        )

        if not os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"):
            # max product of batch size x tkv supported by the Spyre compiler
            default_max_batch_tkv_limit = (
                vllm_config.model_config.max_model_len * vllm_config.scheduler_config.max_num_seqs
            )

            os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = str(default_max_batch_tkv_limit)
            logger.info(
                "No model / tensor parallel size specific value for VLLM_DT_MAX_BATCH_TKV_LIMIT "
                "found. Using the default value (max_model_len * max_batch_size): %d",
                default_max_batch_tkv_limit,
            )
            cls._max_batch_tkv_limit = default_max_batch_tkv_limit
        else:
            cls._set_batch_tkv_limit_from_env()

        handle_disable_compilation(vllm_config, is_decoder)

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Spyre.")
        return False

    @classmethod
    def inference_mode(cls):
        """
        Spyre does not support `torch.inference_mode`.
        This allows to fall back to `torch.no_grad` when inference mode is set.
        """
        return torch.no_grad()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    @classmethod
    def get_warmup_shapes(cls, scheduler_config) -> tuple[dict[str, int], ...]:
        assert scheduler_config.runner_type == "pooling"
        if cls._warmup_shapes is not None:
            return cls._warmup_shapes
        # load warmup shapes and sort by "speed"
        wup_prompt_lens = envs_spyre.SENDNN_INFERENCE_WARMUP_PROMPT_LENS or []
        if not all(pl % 64 == 0 for pl in wup_prompt_lens):
            raise RuntimeError(
                "All values in SENDNN_INFERENCE_WARMUP_PROMPT_LENS must be multiples of 64."
            )

        wup_batch_sizes = envs_spyre.SENDNN_INFERENCE_WARMUP_BATCH_SIZES or []
        if len(wup_prompt_lens) != len(wup_batch_sizes):
            raise RuntimeError(
                "The lists in SENDNN_INFERENCE_WARMUP_PROMPT_LENS and "
                "SENDNN_INFERENCE_WARMUP_BATCH_SIZES must have equal length"
            )

        logger.info("SENDNN_INFERENCE_WARMUP_PROMPT_LENS = %s", wup_prompt_lens)
        logger.info("SENDNN_INFERENCE_WARMUP_BATCH_SIZES = %s", wup_batch_sizes)

        cls._warmup_shapes = tuple(
            sorted(
                [
                    {"prompt_length": pl, "batch_size": bs}
                    for pl, bs in zip(wup_prompt_lens, wup_batch_sizes)
                ],
                key=operator.itemgetter("batch_size", "prompt_length"),
            )
        )
        return cls._warmup_shapes

    @classmethod
    def get_block_size(cls) -> int:
        return cls._block_size

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return True

    @classmethod
    def validate_request(
        cls,
        processed_inputs: "EngineInput",
        params: "SamplingParams | PoolingParams",
    ) -> None:
        """Raises if this request is unsupported on this platform"""

        # The PoolingParams import is lazy here because it imports vllm.config,
        # which will in turn import this file again.
        from vllm.pooling_params import PoolingParams

        if isinstance(params, PoolingParams):
            # Only validating generation requests for now
            return None

        if params.prompt_logprobs is not None:
            raise ValueError("Prompt logprobs are currently not supported.")

        if "encoder_prompt" in processed_inputs:
            raise ValueError("Encoder-decoder models not supported ")
        if "prompt_token_ids" not in processed_inputs:
            # Can't do any extra validation on embedding-only inputs
            return
        prompt_len = len(cast(TokensInput, processed_inputs)["prompt_token_ids"])

        max_tokens = 0
        if params is not None and params.max_tokens is not None:
            max_tokens = params.max_tokens

        # For continuous batching, check if the request is within the max
        # context length. This needs to take the padded prompt length
        # into account.

        # ceil division to pad to next block boundary
        prompt_padding_len = math.ceil(prompt_len / cls._block_size) * cls._block_size
        if prompt_padding_len + max_tokens > cls._config.model_config.max_model_len:
            raise ValueError(
                "Could not add request: prompt length is "
                f"{prompt_len} tokens, which gets padded to "
                f"{prompt_padding_len} tokens, maximum number of output "
                f"tokens is {max_tokens} tokens, but max model context "
                f"length is {cls._config.model_config.max_model_len}."
            )

    @classmethod
    def _get_matching_warmup_shapes(
        cls, prompt_len: int, warmup_shapes: tuple[dict[str, int], ...]
    ) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request (pooling models only)"""
        return [shape for shape in warmup_shapes if prompt_len <= shape["prompt_length"]]

    # Defined here for testing purposes
    DEFAULT_CHUNK_SIZE = 512

    @classmethod
    def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
        if parser is None:
            return

        parser.set_defaults(enable_prefix_caching=True)
        parser.set_defaults(max_num_batched_tokens=cls.DEFAULT_CHUNK_SIZE)
        parser.set_defaults(
            enable_chunked_prefill=True
        )  # set to pass vllm scheduler's max_model_len check

        # Register conditional defaults that apply globally
        ConditionalDefaultManager.register(
            dest="config_format",
            compute_default=_compute_config_format,
        )
        ConditionalDefaultManager.register(
            dest="tokenizer_mode",
            compute_default=_compute_config_format,
        )

        # Apply the conditional default patches to this parser
        # This replaces the actions for managed arguments and patches
        # the base ArgumentParser.parse_args method
        ConditionalDefaultManager.apply(parser)

    @classmethod
    def _check_threading_config(cls, worker_count: int):
        """
        Check parallelism configuration to avoid CPU contention

        Libraries that support multi-threading (eg. OpenMP) default to
        parallelism based on the number of CPUs on the host. This can lead to
        CPU contention in containerized deployments especially when process
        forking is involved. This function provides better default behavior.
        """

        # The quay.io/ibm-aiu/spyre-base image includes shell scripts that
        # automatically set OMP_NUM_THREADS to the result of `nproc --all`.
        #
        # vLLM also already has logic around threading to be aware of,
        #  - sets TORCHINDUCTOR_COMPILE_THREADS=1 (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/env_override.py#L38-L39)
        #  - it will set OMP_NUM_THREADS=1 when using multiple workers (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/executor/multiproc_worker_utils.py#L304)
        #  - has configurations for OMP thread binding (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/envs.py#L435-L438)
        #    - the bind attempts to detect NUMA nodes (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/v1/worker/cpu_worker.py#L111)

        assert worker_count > 0
        # Always print current env for awareness
        env_map = {env: os.getenv(env) for env in THREADING_ENVS}
        logger.info(
            "Initial threading configurations: %s",
            " ".join([f"{env}={value}" for env, value in env_map.items()]),
        )

        # Try to determine the CPU time/cores that we are allocated
        cpu_count: float | None = None
        detection_message = ""

        if (num_cpu := envs_spyre.SENDNN_INFERENCE_NUM_CPUS) > 0:
            cpu_count = num_cpu
            detection_message = f"SENDNN_INFERENCE_NUM_CPUS is set to {cpu_count}"
        else:
            try:
                # try to query cgroup CPU limits
                with open("/sys/fs/cgroup/cpu.max") as f:
                    quota_str, period_str = f.read().strip().split()

                if quota_str != "max":
                    quota = int(quota_str)
                    period = int(period_str)
                    cpu_count = quota / period
                    detection_message = f"Detected cgroup CPU limit of {cpu_count}"

            except FileNotFoundError:
                # file may not exist if not running under cgroups v2
                pass
            except Exception as e:
                logger.debug("Error parsing /sys/fs/cgroup/cpu.max to get CPU info", exc_info=e)

            # try psutil to get physical core count
            if cpu_count is None:
                try:
                    import psutil

                    cpu_count = float(psutil.cpu_count(logical=False))
                    detection_message = (
                        f"Detected {cpu_count} physical CPUs from psutil.cpu_count(logical=False)"
                    )
                except ImportError:
                    logger.info("Install psutil to count physical CPU cores")
                    pass
                except Exception as e:
                    logger.debug("Error using psutil", exc_info=e)

            # could try `nproc` here, but it is affected by
            # OMP_NUM_THREADS itself

            # try os.cpu_count() to get node CPU count
            if cpu_count is None and (cpu_count_res := os.cpu_count()) is not None:
                cpu_count = float(cpu_count_res)
                detection_message = f"Detected {cpu_count} CPUs from `os.cpu_count()`"

        # NOTE: math.ceil can output a number for each worker that sums
        # to a total greater than cpu_count.
        cpus_per_worker = math.ceil(cpu_count / worker_count) if cpu_count is not None else None

        thread_warning = (
            "Excessive threads may result in CPU contention. "
            + "Note that each worker processes has its own thread pools."
            if worker_count > 1
            else ""
        )
        failed_detection_message = (
            "Unable to detect available CPUs to validate threading configuration."
        )

        if envs_spyre.SENDNN_INFERENCE_UPDATE_THREAD_CONFIG:
            if cpus_per_worker is None:
                raise RuntimeError(
                    f"{failed_detection_message} Set SENDNN_INFERENCE_NUM_CPUS or "
                    "use SENDNN_INFERENCE_UPDATE_THREAD_CONFIG=0 and configure "
                    "manually."
                )

            for env in THREADING_ENVS:
                os.environ[env] = str(cpus_per_worker)

            logger.info(
                "%s for %d workers. Since SENDNN_INFERENCE_UPDATE_THREAD_CONFIG is enabled, "
                "setting threading configurations to %d",
                detection_message,
                worker_count,
                cpus_per_worker,
            )
            return

        # In the case that SENDNN_INFERENCE_UPDATE_THREAD_CONFIG is not enabled,
        # check configs and maybe log a warning
        if cpus_per_worker is None:
            logger.info("%s %s", failed_detection_message, thread_warning)
            return

        def _float_or_0(s: str) -> float:
            try:
                return float(s)
            except ValueError:
                return 0.0

        if any(
            (value is None or _float_or_0(value) > 1.2 * cpus_per_worker)
            for value in env_map.values()
        ):
            logger.warning(
                "%s %s for %d workers. Recommend setting each threading configuration to %d. Set "
                "SENDNN_INFERENCE_UPDATE_THREAD_CONFIG=1 to do this automatically.",
                thread_warning,
                detection_message,
                worker_count,
                cpus_per_worker,
            )

    def get_max_output_tokens(self, prompt_len: int) -> int:
        """Return the size of biggest ```new_tokens``` of the \
            warmup shapes that fits the prompt length"""
        if self._warmup_shapes is None:
            # ceil division to pad to next block boundary
            padded_prompt_len = math.ceil(prompt_len / self._block_size) * self._block_size
            max_new_tokens = self._config.model_config.max_model_len - padded_prompt_len
            return max_new_tokens

        max_new_tokens = 1
        for shape in self._warmup_shapes:
            if prompt_len <= shape["prompt_length"]:
                max_new_tokens = max(max_new_tokens, shape["new_tokens"])

        return max_new_tokens

    @classmethod
    def _patch_tokenizer_registry_get_config(cls) -> None:
        """Patch get_config to suppress KeyError when called from tokenizer registry.

        The tokenizer registry imports get_config via:
            from vllm.transformers_utils.config import get_config

        This creates a local reference, so we must patch the registry module's
        reference directly, not just the source module.

        """
        import vllm.tokenizers.registry as tokenizer_registry

        original_get_config = tokenizer_registry.get_config

        def safe_get_config(*args, **kwargs):
            try:
                return original_get_config(*args, **kwargs)
            except KeyError:
                return None

        # Patch the imported reference in the registry module
        tokenizer_registry.get_config = safe_get_config  # type:ignore[invalid-assignment]

        logger.debug("Patched get_config in vllm.tokenizers.registry to suppress KeyError")

    @classmethod
    def is_backend_sendnn_enabled(cls) -> bool:
        return envs_spyre.SENDNN_INFERENCE_DYNAMO_BACKEND in ("sendnn", "sendnn_compile_only")

    @classmethod
    def maybe_ensure_sendnn_configured(cls, model_config: ModelConfig) -> None:
        """If using sendnn, import torch_sendnn and check configuration.

        If torch_sendnn is imported too early, it may have the wrong
        configuration. An assertion error will be raised in this case. This
        function must be called before triggering any torch compilation.
        """
        if not cls._torch_sendnn_configured and cls.is_backend_sendnn_enabled():
            try:
                import torch_sendnn  # ty: ignore[unresolved-import] # noqa: F401
            except ImportError as err:
                raise RuntimeError("sendnn backend requires torch_sendnn") from err

            # We only require checks for the `VLLM_DT_*` environment variables that need to be set
            # at torch_sendnn import time for generative models.
            if model_config.runner_type != "generate":
                cls._torch_sendnn_configured = True
                return

            # If the compilation cache is disabled, then we cannot check any of the config from
            # torch_sendnn directly
            if not bool(int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0"))):
                cls._torch_sendnn_configured = True
                return

            # TODO: This is a hack to make sure that the sendnn backend is
            # configured correctly. Environment variables are captured at
            # import time, so we assert that values were captured with the
            # values we set
            # NB: must use getattr due to Python name mangling
            try:
                sendnn_backend_state = getattr(torch_sendnn.backends.sendnn_backend, "__state")
                actual_config = sendnn_backend_state.spyre_graph_cache.deeptools_config["config"]
            except (AttributeError, KeyError) as e:
                logger.warning(
                    "Error reading torch_sendnn backend state for validation: %s", str(e)
                )
                # Let this fall through and log many warnings to be noisy
                actual_config = {}

            # Validate environment variables and config values match
            env_to_config = {
                "VLLM_DT_CHUNK_LEN": "vllm_chunk_length",
                "VLLM_DT_MAX_CONTEXT_LEN": "vllm_max_context_length",
                "VLLM_DT_MAX_BATCH_SIZE": "vllm_max_batch_size",
                "VLLM_DT_MAX_BATCH_TKV_LIMIT": "vllm_max_batch_tkv_limit",
            }

            # Intentionally noisy logging for increased visibility
            backend_state_looks_valid = True
            for env_var, config_key in env_to_config.items():
                actual = actual_config.get(config_key)
                expected = os.getenv(env_var)
                if actual is None:
                    logger.warning(
                        "torch_sendnn may be misconfigured! %s does not exist as expected",
                        config_key,
                    )
                    backend_state_looks_valid = False

                if expected is None:
                    logger.warning("%s must be set before importing torch_sendnn", env_var)
                    backend_state_looks_valid = False

                if actual != expected:
                    logger.warning(
                        "torch_sendnn is misconfigured! %s: expected '%s', got '%s'",
                        config_key,
                        expected,
                        actual,
                    )
                    backend_state_looks_valid = False

            assert backend_state_looks_valid, (
                "torch_sendnn backend state could not be validated! Please "
                "report this issue to maintainers."
            )
            cls._torch_sendnn_configured = True

    @classmethod
    def _set_batch_tkv_limit_from_env(cls) -> None:
        try:
            cls._max_batch_tkv_limit = int(os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT", "-1"))  #  ty: ignore
        except ValueError as e:
            raise ValueError("VLLM_DT_MAX_BATCH_TKV_LIMIT must be an integer") from e


def _compute_config_format(namespace: argparse.Namespace) -> str:
    """Check if a model is in mistral format by looking for params.json.

    This uses any_pattern_in_repo_files which correctly handles both local paths
    and HuggingFace cache, including offline mode support.
    """
    from vllm.transformers_utils.repo_utils import any_pattern_in_repo_files, get_model_path

    # Check both 'model' and 'model_tag' since vLLM uses different
    # attribute names in different contexts
    model = getattr(namespace, "model_tag", None) or getattr(namespace, "model", "") or ""

    if not model:
        return "auto"

    # Get optional HF arguments
    revision = getattr(namespace, "revision", None)
    token = getattr(namespace, "hf_token", None)

    # Resolve local path in offline mode (if not already a local path)
    if huggingface_hub.constants.HF_HUB_OFFLINE:
        model = get_model_path(model, revision)

    # Look for params.json which indicates a mistral-format model
    if any_pattern_in_repo_files(
        model,
        allow_patterns=["params.json"],
        revision=revision,
        token=token,
    ):
        return "mistral"
    return "auto"


# 🌶️🌶️🌶️ Patch vllm.tokenizers.registry to suppress KeyError from get_config
# The tokenizer registry calls get_config() which can raise KeyError when
# an unknown model_type is encountered in LazyConfigDict. The original code
# only suppresses ValueError and OSError, but KeyError should also be
# suppressed since it's expected for models not in the registry.
# This must be done at import time to be applied to spawned worker processes.
SpyrePlatform._patch_tokenizer_registry_get_config()
