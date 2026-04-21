"""A Spyre worker class."""

import contextlib
import functools
import json
import os
import platform
import signal
import sys
import time
import math
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

import torch
import torch.distributed as dist
import vllm.envs as envs
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

import sendnn_inference.envs as envs_spyre
import sendnn_inference.perf_metrics as perf_metrics
import sendnn_inference.utils as utils_spyre
from sendnn_inference.model_executor.model_loader import spyre_setup
from sendnn_inference.platform import SpyrePlatform
from sendnn_inference.v1.worker.spyre_model_runner import (
    ChunkedPrefillModelRunner,
    SpyrePoolingModelRunner,
    SupportedTask,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput


logger = init_logger(__name__)

# var to make sure we always warmup with the right context
_inside_warmup_mode = False


def new_request_data_builder(
    req_id: str,
    block_ids: tuple[list[int]],
    prompt_token_ids: list[int],
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
    prompt_embeds: torch.Tensor | None,
    mm_features: list | None,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
        mm_features=mm_features or [],
        prompt_embeds=prompt_embeds,
    )


@contextlib.contextmanager
def _maybe_warmup_context(limit: int, world_size: int, rank: int):
    global _inside_warmup_mode
    warmup_context = contextlib.nullcontext
    if SpyrePlatform.is_backend_sendnn_enabled():
        from torch_sendnn import warmup_mode  # ty: ignore

        warmup_context = warmup_mode

    sendnn_exit = warmup_context.__exit__

    # We wrap warmup_mode's __exit__ method in a stagger region
    # as it's where the model is actually compiled.
    def __stagger_exit__(*args, **kwargs):
        with utils_spyre.stagger_region(limit, world_size, rank):
            sendnn_exit(*args, **kwargs)

    # Use update_wrapper to make __stagger_exit__ look like a proper instance
    # method on warmup_context
    functools.update_wrapper(__stagger_exit__, sendnn_exit)
    # Replace `warmup_context.__exit__` with our new wrapper
    warmup_context.__exit__ = __stagger_exit__  # type: ignore[method-assign]

    # FIXME: This is now staggering on every forward call, which is not great

    with warmup_context():
        _inside_warmup_mode = True
        yield
        _inside_warmup_mode = False


@contextlib.contextmanager
def use_torch_fx_backed_size_oblivious():
    # this setting is required to mark a dimension of size 1 as dynamic
    # for pytorch >= 2.7.1 (needed to support batch size 1 for decodes)
    # NB: this setting is disabled at the end of this function
    from torch.fx.experimental import _config as config

    config.backed_size_oblivious = True  # ty: ignore[invalid-assignment]
    yield
    config.backed_size_oblivious = False


class SpyreWorker(WorkerBase):
    """A worker class that executes the model on a group of Spyre cores."""

    @property
    def is_pooling(self) -> bool:
        return self.model_config.runner_type == "pooling"

    @property
    def is_decoder(self) -> bool:
        return self.model_config.runner_type == "generate"

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation.

        These specs are used to:
        - build the kv_cache_configs that are then passed to
            initialize_from_config() on this instance
        - determine the number of available kv_cache_blocks, see
            SpyreWorker.determine_available_memory
        """
        return self.model_runner.get_kv_cache_spec()

    def compile_or_warm_up_model(self) -> float:
        """Prepare model for execution through compilation/warmup.

        Returns:
            The accumulated compilation time in seconds.
        """

        if self.is_decoder:
            return self._warmup_spyre_dynamic_size(self.restricted_tokens)
        if self.model_runner.is_multimodal:
            raise NotImplementedError("[WARMUP] multimodal models are not supported yet.")
        num_shape_combinations = len(self.spyre_warmup_shapes)
        logger.info(
            "[WARMUP] Starting for %d prompt/decode/batchsize-shape combinations...",
            len(self.spyre_warmup_shapes),
        )
        all_warmup_start_t = time.time()
        for i, (prompt_len, batch_size) in enumerate(
            [(s["prompt_length"], s["batch_size"]) for s in self.spyre_warmup_shapes]
        ):
            # warmup individual combination
            logger.info(
                "[WARMUP] (%d/%d) for prompt length %d with batch size %d...",
                i + 1,
                num_shape_combinations,
                prompt_len,
                batch_size,
            )
            self._warmup_spyre_fixed_size(prompt_len, self.restricted_tokens, batch_size)

        self.model_runner.complete_warmup()

        all_warmup_end_t = time.time()
        all_warmup_total_t = all_warmup_end_t - all_warmup_start_t
        self.perf_metrics.log("total warmup time", all_warmup_total_t)
        # No more perf metric are captured (so far) after warmup, cleanup now.
        del self.perf_metrics
        logger.info(
            "[WARMUP] All %d prompt/decode/batchsize-shape combinations finished in %.3fs",
            num_shape_combinations,
            all_warmup_total_t,
        )
        return all_warmup_total_t

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        # TODO: Implement something!
        return

    def determine_available_memory(self) -> int:
        """Return available device memory in bytes.

        This is used in conjunction with the result from `get_kv_cache_spec`
        to determine the number of KV cache blocks that can fit on the device.

        The number of available blocks is calculated as:
            available_memory / page_size / # of layers
        where the page size and number of layers come from the kv cache spec.

        The number of device blocks (called "gpu blocks" in most places) can
        also be overridden by `--num-gpu-blocks-override`, which is set under
        `vllm_config.cache_config.num_gpu_blocks_override`.

        🌶️🌶️🌶️ The result from this method _only_ applies to the KV Cache
        management in vLLM's core scheduler. This does _not_ apply to the KV
        cache management handled directly by the sendnn-inference worker and model
        runner. We return a minimal value here to make the vllm scheduler happy.
        """
        # The fake kv_cache config specified by the model runner sets 4 bytes
        # per token.
        # TODO: For multimodal, we may want to explicitly validate that the max
        # len is less than the maximum size of one multimodal object prior to
        # this, otherwise we can see some cryptic behaviors here.
        accurate_fake_kv_cache_size = (
            4 * self.model_config.max_model_len * self.scheduler_config.max_num_seqs
        )

        # The vLLM scheduler reserves a null block in its kv-cache, so we need
        # at least one more block to allow for proper scheduling. We double
        # the cache size here to ensure that the vllm scheduler always has
        # blocks available. This causes the log message from vLLM about it's
        # KV cache capacity to be double the log message from sendnn-inference.
        # This can probably be fixed in a nicer way.
        return 2 * accurate_fake_kv_cache_size

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Construct the KV cache from the provided configs.
        Currently, we do not support paged attention or kv caching"""
        pass

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # For power-user debugging of spyre logs for tensor parallel ops
        self.redirect_logs_to_files()

        self.perf_metrics = perf_metrics.create_perf_metric_logger(rank)
        if self.parallel_config and is_driver_worker:
            assert rank % self.parallel_config.tensor_parallel_size == 0, (
                "Driver worker should be rank 0 of tensor parallel group."
            )
        self.model_runner: Union[
            ChunkedPrefillModelRunner,
            SpyrePoolingModelRunner,
        ]
        self.warmup_block_ids = 1
        if self.is_pooling:
            self.model_runner = SpyrePoolingModelRunner(
                self.vllm_config, self.is_driver_worker, self.rank
            )
            self.spyre_warmup_shapes = SpyrePlatform.get_warmup_shapes(
                self.vllm_config.scheduler_config
            )
        else:
            self.model_runner = ChunkedPrefillModelRunner(
                self.vllm_config, self.is_driver_worker, self.rank
            )

        self._env_initialized = False
        # Torch profiler. Enabled and configured through ProfilerConfig. Set via:
        #   --profiler-config.profiler=torch
        #   --profiler-config.torch_profiler_dir=/path/to/save/trace)
        # OR
        #   --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/save/trace"}'
        self.profiler_config = vllm_config.profiler_config
        self.profiler: TorchProfilerWrapper | None = None
        if self.profiler_config.profiler == "torch" and SpyrePlatform.is_backend_sendnn_enabled():
            logger.info_once(
                "Traces will contain AIU events if PyTorch with AIU profiling support is installed."
            )
            os.environ["ProfilerActivity"] = "PrivateUse1"  # noqa: SIM112

            # Get the current value of DT_OPT and autopilot
            dt_opt = os.environ.get("DT_OPT", "")
            options = dict(opt.split("=") for opt in dt_opt.split(",") if "=" in opt)
            autopilot_opt = options.get("autopilot", "1")  # autopilot defaults to 1 if not set
            if autopilot_opt == "1":
                logger.warning_once(
                    "autopilot on detected with profiling enabled. Add "
                    "autopilot=0 to DT_OPT to see individual AIU-kernel "
                    "execution in the trace."
                )

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

        if SpyrePlatform.is_backend_sendnn_enabled():
            spyre_setup.spyre_dist_setup(
                rank=self.rank, world_size=self.parallel_config.world_size, verbose=True
            )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

    def redirect_logs_to_files(self) -> None:
        """Redirects all stdout and stderr to a rank-specific logfile.

        This is 🌶️🌶️🌶️ for debugging purposes only, after it is invoked there
        won't be any more logs on stdout or stderr from this worker process.
        """
        if envs_spyre.SENDNN_INFERENCE_WORKER_LOG_REDIRECT_DIR:
            log_dir = Path(envs_spyre.SENDNN_INFERENCE_WORKER_LOG_REDIRECT_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"rank-{self.rank}.log"

            logger.warning("Redirecting all logs to %s", str(log_path))

            # As written, this isn't reversible. This could be made into a
            # context manager with cleanup, but this is for debug purposes only.
            # This uses system calls because we need to catch the output of all
            # the C libraries running in this process under the hood.

            # Open the file for redirection
            redirected_file = log_path.open("w")
            redirected_fd = redirected_file.fileno()

            # Redirect stderr and stdout with `dup2`
            os.dup2(redirected_fd, sys.stderr.fileno())
            os.dup2(redirected_fd, sys.stdout.fileno())

    def init_device(self) -> None:
        if platform.machine() == "s390x":
            from torch.serialization import LoadEndianness

            torch.serialization.set_default_load_endianness(LoadEndianness.LITTLE)

        if not self._env_initialized:
            backend = "gloo"
            init_method = "env://"

            init_distributed_environment(
                world_size=self.parallel_config.world_size,
                rank=self.rank,
                distributed_init_method=init_method,
                backend=backend,
                timeout=timedelta(minutes=envs_spyre.SENDNN_INFERENCE_GLOO_TIMEOUT_MINUTES),
            )

            if self.parallel_config.world_size > 1:
                self.init_distributed_environment()
            elif SpyrePlatform.is_backend_sendnn_enabled():
                spyre_setup.spyre_setup()

            ensure_model_parallel_initialized(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.pipeline_parallel_size,
            )

            self._env_initialized = True

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        assert self._env_initialized

        is_local = os.path.isdir(self.model_config.model)
        if is_local:
            cf_file = os.path.join(self.model_config.model, "config.json")
        else:
            cf_file = hf_hub_download(
                repo_id=self.model_config.model,
                revision=self.model_config.revision,
                filename="config.json",
            )
        with open(cf_file, "rb") as f:
            config = json.load(f)

        restricted_tokens = []
        if tok := config.get("bos_token_id") is not None:
            restricted_tokens.append(int(tok))
        if tok := config.get("eos_token_id") is not None:
            restricted_tokens.append(int(tok))

        self.restricted_tokens = restricted_tokens

        logger.info("load model...")
        # TODO: check additionally if the Spyre card has enough memory
        # for all requested model warmups
        # printing env variables for debugging purposes
        load_model_start_t = time.time()
        self.model_runner.load_model()
        load_model_end_t = time.time()

        load_model_total_t = load_model_end_t - load_model_start_t
        self.perf_metrics.log("load model time", load_model_total_t, model=self.model_config.model)
        logger.info("load model took %.3fs", load_model_total_t)

    def _gen_warmup_block_ids(self, num_tokens: int) -> tuple[list[int]]:
        num_blocks = math.ceil(num_tokens / 64)
        start = self.warmup_block_ids
        end = start + num_blocks
        self.warmup_block_ids = end
        return ([i for i in range(start, end)],)

    def _warmup_spyre_dynamic_size(self, special_token_ids) -> float:
        warmup_start_t = time.time()

        # satisfy mypy
        model_runner: ChunkedPrefillModelRunner = cast(ChunkedPrefillModelRunner, self.model_runner)

        vocab_size = model_runner.vocab_size

        valid_token_ids = [i for i in range(1, vocab_size) if i not in set(special_token_ids)]

        # Convert to tensor for sampling
        valid_token_ids_tensor = torch.tensor(
            valid_token_ids, dtype=torch.long, device=torch.device("cpu")
        )
        num_decode_tokens = 2
        # TODO: we need 2 requests for warmup on FP8+CB
        # Check if model is quantized
        req_count = 3 if self.model_config.quantization is not None else 2

        mm_model_utils = self.model_runner.get_mm_utils()
        if mm_model_utils:
            # In the case of multimodal, delegate to the MM utils class to get
            # the appropriate features; note prompt length is currently
            # determined purely by the multimodal input encoding.
            mm_warmup_inputs = mm_model_utils.get_warmup_inputs(req_count)
            warmup_tokens = mm_warmup_inputs.input_ids
            warmup_embeds_tensor = mm_warmup_inputs.input_embeds
            mm_features = mm_warmup_inputs.mm_features
            prompt_len = len(warmup_tokens[0])
        else:
            prompt_len = 42
            warmup_tokens_tensor = valid_token_ids_tensor[
                torch.randint(0, len(valid_token_ids_tensor), (3, prompt_len))
            ]
            warmup_tokens = [wt.tolist() for wt in warmup_tokens_tensor]
            # Text only models don't use mm features, and currently we only
            # use embeddings as inputs to multimodal models.
            warmup_embeds_tensor = [None] * req_count
            mm_features = None

        requests = [
            new_request_data_builder(
                req_id="warmup-%d" % (i),
                prompt_token_ids=warmup_tokens[i],
                block_ids=self._gen_warmup_block_ids(len(warmup_tokens[i])),
                sampling_params=SamplingParams(max_tokens=num_decode_tokens),
                pooling_params=None,
                prompt_embeds=warmup_embeds_tensor[i],
                mm_features=mm_features,
            )
            for i in range(req_count)
        ]

        warmup_requests = requests[:-1]  # first one or two
        deploy_req = requests[-1]  # Last one
        model_runner.pre_warmup()

        with _maybe_warmup_context(
            envs_spyre.SENDNN_INFERENCE_MAX_LOAD_PROCESSES,
            self.parallel_config.world_size,
            self.rank,
        ):
            # TODO(wallas): I am not sure if really need warmup with at
            # least batch size 2 for quantized model
            self._dynamic_warmup(
                requests=warmup_requests,
                prompt_len=prompt_len,
                valid_token_ids_tensor=valid_token_ids_tensor,
            )

        # warmup_mode completes the graph compilation, but we need to do
        # one additional prefill to deploy the compiled program to the device,
        # the necessary operations are included in the graph and will be removed
        # after this execution

        # update sampling_params here to ensure logits processing code is also
        # compiled during warmup
        deploy_req.sampling_params = SamplingParams(
            temperature=1.0,
            top_k=10,
            top_p=0.9,
            min_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            repetition_penalty=1.2,
            max_tokens=4,
            min_tokens=1,
            logprobs=1,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[deploy_req],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={deploy_req.req_id: prompt_len},
            total_num_scheduled_tokens=prompt_len,
            finished_req_ids=set(),
            **_get_extra_args(),
        )
        logger.info("[WARMUP] Deploying to device...")
        self.execute_model(scheduler_output)
        self._cleanup_model_runner(request=[deploy_req])

        model_runner.complete_warmup()

        warmup_end_t = time.time()
        warmup_total_t = warmup_end_t - warmup_start_t
        compile_cache_str = (
            "enabled" if int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0")) else "disabled"
        )
        logger.info(
            "[WARMUP] Finished in %.3fs (compilation cache %s)", warmup_total_t, compile_cache_str
        )

        maybe_override_signals_handler()
        return warmup_total_t

    def _cleanup_model_runner(self, request) -> None:
        # Needed to clean up the data of model runner
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            # NOTE: this means no work to do
            total_num_scheduled_tokens=0,
            # The requests to be removed
            finished_req_ids=set([r.req_id for r in request]),
            **_get_extra_args(),
        )
        self.execute_model(scheduler_output)
        # satisfy mypy
        model_runner: ChunkedPrefillModelRunner = cast(ChunkedPrefillModelRunner, self.model_runner)
        model_runner.tkv = 0

    def _warmup_spyre_fixed_size(self, prompt_len, special_token_ids, batch_size):
        assert self.is_pooling, "only pooling models have fixed warmup shapes"
        warmup_start_t = time.time()
        # NOTE(ngl): empty tensor causes spyre to hang, so using
        # randint without 0 and the eos and bos token

        # Create a list of valid values between 1 (inclusive) and vocab
        # size (exclusive) by excluding the eos and bos token ids
        # (in special_token_ids)
        vocab_size = self.model_runner.vocab_size
        valid_token_ids = [i for i in range(1, vocab_size) if i not in set(special_token_ids)]
        # Convert to tensor for sampling
        valid_token_ids_tensor = torch.tensor(
            valid_token_ids, dtype=torch.long, device=torch.device("cpu")
        )

        # Sample from the valid token ids
        warmup_tokens_tensor = valid_token_ids_tensor[
            torch.randint(0, len(valid_token_ids_tensor), (batch_size, prompt_len))
        ]

        sampling_params, pooling_params = None, None

        pooling_params = PoolingParams(task="embed")  # for warmup any task will do

        # Set up dummy requests for prefill steps
        dummy_requests = [
            new_request_data_builder(
                req_id=f"warmup_{i}",
                prompt_token_ids=warmup_tokens_tensor[i].tolist(),
                block_ids=self._gen_warmup_block_ids(len(warmup_tokens_tensor[i])),
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                prompt_embeds=None,
                mm_features=None,
            )
            for i in range(batch_size)
        ]

        # Set up dummy cached_requests for decode steps
        req_ids = []
        new_token_ids = []
        new_block_ids = []
        num_computed_tokens = []
        for req in dummy_requests:
            req_ids.append(req.req_id)
            new_token_ids.append(
                [valid_token_ids_tensor[torch.randint(0, len(valid_token_ids_tensor), (1,)).item()]]  # ty: ignore
            )  # placeholder token
            new_block_ids.append([req.block_ids])
            num_computed_tokens.append(req.num_computed_tokens)

        cached_request_data = CachedRequestData.make_empty()
        cached_request_data.req_ids = req_ids
        cached_request_data.new_block_ids = new_block_ids
        cached_request_data.new_token_ids = new_token_ids
        cached_request_data.num_computed_tokens = num_computed_tokens

        # Set up scheduler_output for execute_model
        for r in dummy_requests:
            assert r.prompt_token_ids is not None
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=dummy_requests,
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={r.req_id: self._get_num_tokens(r) for r in dummy_requests},
            total_num_scheduled_tokens=sum(prompt_len for _ in range(batch_size)),
            finished_req_ids=set(),
            **_get_extra_args(),
        )

        # First full forward pass
        logger.info("[WARMUP] Compiling graphs...")
        # The fixed size warmup needs to happen only in here
        with _maybe_warmup_context(
            envs_spyre.SENDNN_INFERENCE_MAX_LOAD_PROCESSES,
            self.parallel_config.world_size,
            self.rank,
        ):
            self._warmup_model_forward_pass(scheduler_output, dummy_requests, cached_request_data)
        self.perf_metrics.log(
            "warmup 1 time",
            time.time() - warmup_start_t,
            batch_size=batch_size,
            prompt_len=prompt_len,
        )

        # Second full forward pass
        logger.info("[WARMUP] Deploying to device...")
        warmup2_start_t = time.time()
        self._warmup_model_forward_pass(scheduler_output, dummy_requests, cached_request_data)

        warmup_end_t = time.time()
        warmup_total_t = warmup_end_t - warmup_start_t
        self.perf_metrics.log(
            "warmup 2 time",
            time.time() - warmup2_start_t,
            batch_size=batch_size,
            prompt_len=prompt_len,
        )
        compile_cache_str = (
            "enabled" if int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0")) else "disabled"
        )
        logger.info(
            "[WARMUP] Prompt length %d finished in %.3fs (compilation cache %s)",
            prompt_len,
            warmup_total_t,
            compile_cache_str,
        )
        maybe_override_signals_handler()

    @use_torch_fx_backed_size_oblivious()
    def _dynamic_warmup(
        self,
        requests: list[NewRequestData],
        prompt_len: int,
        valid_token_ids_tensor: torch.Tensor,
    ) -> None:
        # TODO: because of FP8 we are doing warmup with bs=2 again.
        # Once we figure it out this limitation we should revert this to
        # bs=1 again.
        assert _inside_warmup_mode, "it looks like you are outside the warmup context for warmup"

        req_count = len(requests)
        for idx, req in enumerate(requests):
            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=[req],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={req.req_id: prompt_len},
                total_num_scheduled_tokens=prompt_len,
                finished_req_ids=set(),
                **_get_extra_args(),
            )

            logger.info("[WARMUP] Prefill [%s/%s]...", idx + 1, req_count)

            self.execute_model(scheduler_output)

        random_token_id = lambda: torch.randint(0, len(valid_token_ids_tensor), (1,)).item()

        cached_request_data = CachedRequestData.make_empty()
        cached_request_data.req_ids = [req.req_id for req in requests]
        cached_request_data.new_block_ids = []
        for req in requests:
            if len(req.prompt_token_ids) % 64 == 0:
                cached_request_data.new_block_ids.append(self._gen_warmup_block_ids(1))
            else:
                cached_request_data.new_block_ids.append(([],))
        cached_request_data.new_token_ids = [
            [valid_token_ids_tensor[random_token_id()]] for _ in requests
        ]
        cached_request_data.num_computed_tokens = [prompt_len for _ in requests]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={req.req_id: 1 for req in requests},
            total_num_scheduled_tokens=1,
            finished_req_ids=set(),
            **_get_extra_args(),
        )
        logger.info("[WARMUP] Decode...")
        self.execute_model(scheduler_output)
        self._cleanup_model_runner(request=requests)

    def _warmup_model_forward_pass(
        self,
        scheduler_output: SchedulerOutput,
        requests: list[NewRequestData],
        cached_request_data: CachedRequestData,
    ):
        """Handle a complete forward pass"""
        assert self.is_pooling, "only pooling models have fixed warmup shapes"
        scheduler_output.scheduled_new_reqs = requests
        scheduler_output.scheduled_cached_reqs = CachedRequestData.make_empty()
        scheduler_output.num_scheduled_tokens = {
            r.req_id: self._get_num_tokens(r) for r in requests
        }
        self.execute_model(scheduler_output)  # Prefill

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )
        if is_start:
            # Recreate the wrapper each time
            worker_name = f"{self.vllm_config.instance_id}-rank-{self.rank}"
            if profile_prefix:
                worker_name = f"{profile_prefix}_{worker_name}"
            self.profiler = TorchProfilerWrapper(
                self.profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )
            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    @property
    def do_metadata_broadcast(self) -> bool:
        return True

    @property
    def kv_cache(self) -> list[list[torch.Tensor]] | None:
        return None

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def sample_tokens(self, grammar_output: "GrammarOutput | None") -> ModelRunnerOutput:
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        return EMPTY_MODEL_RUNNER_OUTPUT

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        if self.profiler is not None:
            self.profiler.step()
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def _get_num_tokens(self, r: NewRequestData) -> int:
        assert r.prompt_token_ids is not None, "requests should have tokens!"
        return len(r.prompt_token_ids)


# Ref: https://github.com/vllm-project/vllm/blob/5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7/vllm/v1/executor/multiproc_executor.py#L446 # noqa: E501
# TODO: review this in the future
# This setup is a workaround to suppress logs that are dumped at the shutdown
# of the engine (only on V1) when vllm runs with multiprocess. The undesired
# behavior happens because g3log from Spyre runtime overrides the signal
# handler from vLLM when it starts a process for the engine code. Therefore,
# the engine does not have a chance to gracefully shutdown.
def maybe_override_signals_handler():
    if not (
        envs.VLLM_ENABLE_V1_MULTIPROCESSING and envs_spyre.SENDNN_INFERENCE_OVERRIDE_SIGNALS_HANDLER
    ):
        return

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def _get_extra_args() -> dict:
    """Add any required backwards compatibility code for constructing
    SchedulerOutputs here"""
    return {
        "free_encoder_mm_hashes": [],
        "scheduled_spec_decode_tokens": {},
        "scheduled_encoder_inputs": {},
        "num_common_prefix_blocks": [],
    }
