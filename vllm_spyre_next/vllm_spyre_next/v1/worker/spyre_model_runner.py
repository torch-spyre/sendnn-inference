"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

Data flow:
- self.device = CPU. Buffers and scatter ops stay on CPU.
- _SpyreModelWrapper converts input_ids/positions to Spyre int64 at the
  model call boundary.
- Embedding: Spyre int64 input → Spyre compute → float16 output on Spyre.
- Hidden states flow on Spyre between decoder layers.
- Attention block: Spyre input → CPU compute → Spyre output.
- _SpyreModelWrapper converts final hidden_states to CPU for downstream
  operations (logits indexing, lm_head, sampling).

Float .gpu buffers are on Spyre (via _make_buffer / SpyreCpuGpuBuffer).
Int/bool .gpu buffers stay on CPU and are are aliased to .cpu (CPUModelRunner pattern).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_spyre_next.custom_ops.utils import convert

logger = init_logger(__name__)


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """CpuGpuBuffer with Spyre-safe copies and split dtypes.

    For float dtypes: .cpu on CPU, .gpu on Spyre (float16).
    For int/bool dtypes: .gpu aliased to .cpu (CPUModelRunner pattern).
    All copies are synchronous (Spyre doesn't support non_blocking).
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        cpu_dtype: torch.dtype,
        gpu_dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=cpu_dtype, device="cpu", pin_memory=pin_memory)
        if device.type == "spyre":
            self.gpu = torch.zeros(*size, dtype=gpu_dtype, device=device)
        else:
            # int/bool: alias gpu = cpu (CPUModelRunner pattern)
            self.gpu = self.cpu
        self.np: np.ndarray
        if with_numpy:
            if cpu_dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call SpyreCpuGpuBuffer with "
                    "with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.gpu if n is None else self.gpu[:n]
        src = self.cpu if n is None else self.cpu[:n]
        dst = self.gpu if n is None else self.gpu[:n]
        dst.copy_(src)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.cpu if n is None else self.cpu[:n]
        src = self.gpu if n is None else self.gpu[:n]
        dst = self.cpu if n is None else self.cpu[:n]
        cpu_src = convert(src, device="cpu")
        dst.copy_(cpu_src)
        return dst


class _SpyreModelWrapper:
    """Transparent wrapper that converts model inputs/outputs at the boundary.

    Input conversion (CPU → Spyre):
        input_ids and positions arrive as CPU tensors (int32/int64) because
        self.device=CPU in the runner and buffer scatter ops run on CPU.
        Convert them to Spyre int64 here so the embedding layer takes the
        Spyre fast-path and hidden_states flow on Spyre from the start.

    Output conversion (Spyre → CPU):
        The model's final hidden_states come out on Spyre. Downstream
        operations (indexing via logits_indices, compute_logits/lm_head,
        sampling) all run on CPU.

    Wrapping at the model level ensures ALL call sites get the right
    device — both execute_model (via _model_forward) and _dummy_run
    (which calls self.model(...) directly).
    """

    def __init__(self, model: nn.Module, spyre_device: torch.device):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)

    def __call__(self, *args, **kwargs):
        # Convert integer tensor inputs to Spyre int64
        for key in ("input_ids", "positions"):
            val = kwargs.get(key)
            if val is not None:
                kwargs[key] = convert(val, dtype=torch.int64, device=self._spyre_device)

        result = self._model(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            if result.device.type == "spyre":
                return convert(result, device="cpu")
            return result
        if isinstance(result, tuple):
            return tuple(
                convert(t, device="cpu")
                if isinstance(t, torch.Tensor) and t.device.type == "spyre"
                else t
                for t in result
            )
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for Spyre.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre for floats, aliased to CPU for int/bool

    Inherits from GPUModelRunner to preserve
    the dual-buffer device placement pattern.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Store the real Spyre device before super().__init__ so that
        # _make_buffer can place .gpu tensors on Spyre directly.
        self._spyre_device = device

        # Check if the model dtype is different from float16,
        # which is only currently supported in torch-spyre
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError(
                f"The model dtype needs to be torch.float16 for spyre, "
                f"but was specified to be {vllm_config.model_config.dtype}"
            )

        # Phase 1: Init with device="cpu" to avoid dtype/device errors.
        # Many components create tensors on self.device during init, and
        # Spyre doesn't support all dtypes (int32, bool) natively.
        # _make_buffer (overridden below) already places .gpu on Spyre
        # via self._spyre_device regardless of self.device.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Keep self.device as CPU so buffer management (scatter, copy) stays
        # on CPU. _SpyreModelWrapper converts input_ids/positions to Spyre
        # int64 at the model call boundary, so the embedding takes the Spyre
        # fast-path and hidden_states flow on Spyre between decoder layers.
        # _make_buffer (overridden below) places float .gpu tensors on Spyre
        # regardless of self.device.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernel with C++ CPU implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # Same replacement as CPUModelRunner._postprocess_triton().
        import vllm.utils.cpu_triton_utils as cpu_tl
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._compute_slot_mapping_kernel = cpu_tl.compute_slot_mapping_kernel

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load model and compile for Spyre."""
        logger.info("Loading model %s...", self.model_config.model)

        # Load model on CPU
        self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
        # Attention is nn.Module (not PluggableLayer) so OOT registration is
        # not possible. Patch _apply to no-op before model.to("spyre") so
        # the CPU attention backend can access scale buffers without device
        # mismatch.
        from vllm.model_executor.layers.attention.attention import Attention

        for module in self.model.modules():
            if isinstance(module, Attention):
                module._apply = lambda fn, recurse=True, _m=module: _m

        # Move layer weights to Spyre device.
        # SpyreCpuFallbackMixin._apply() no-op keeps CPU fallback layer
        # weights on CPU (linear, embedding, rotary, lm_head).
        # Spyre-native layers (RMSNorm, SiluAndMul) get their weights moved.
        self.model.to(device=self._spyre_device)
        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Wrap model so ALL forward() calls to the entire model,
        # for example in execute_model, _dummy_run, etc.,
        # automatically convert Spyre outputs to CPU. This ensures downstream
        # indexing (logits_indices), lm_head (CPU weights), and sampling all
        # receive CPU tensors without needing per-call-site overrides.
        self.model = _SpyreModelWrapper(self.model, self._spyre_device)

        logger.info("Model loaded and compiled for Spyre.")

    def _compile_for_spyre(self) -> None:
        """Apply torch.compile for Spyre with static shapes.

        Spyre compilation is handled here (not by vLLM's @support_torch_compile)
        because Spyre requires static shapes — dynamic shapes (SymInt) are not
        supported by the Spyre Inductor backend.

        Supported modes:
        - enforce_eager=True: no compilation (eager execution)
        - CompilationMode.NONE: Spyre-managed compilation with torch.compile
        Other vLLM compilation modes (VLLM_COMPILE, STOCK_TORCH_COMPILE) are
        not supported — the platform forces CompilationMode.NONE in
        apply_config_platform_defaults().
        """
        from vllm.config import CompilationMode

        mode = self.compilation_config.mode
        if mode != CompilationMode.NONE:
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. "
                f"Only CompilationMode.NONE is supported. Spyre handles "
                f"compilation internally via _compile_for_spyre(). "
                f"Use enforce_eager=True to disable compilation entirely."
            )

        if self.vllm_config.model_config.enforce_eager:
            logger.info("Compilation disabled (enforce_eager=True)")
            return

        # Custom ops (spyre_rmsnorm, spyre_cpu_fallback, etc.) are opaque
        # to dynamo but don't cause graph breaks — fullgraph=True is safe.
        # dynamic=False ensures static shapes (Spyre can't handle SymInt).
        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        logger.info("Model compiled for Spyre (backend=inductor)")

    def warming_up_model(self) -> None:
        """Run a dummy forward pass to warm up the model.

        _dummy_run creates CPU int inputs, but _SpyreModelWrapper converts
        input_ids/positions to Spyre int64 at the model boundary. The
        embedding thus runs on Spyre and hidden_states flow on Spyre.
        _SpyreModelWrapper also converts final outputs back to CPU.

        When enforce_eager=False, this also triggers torch.compile.
        """
        logger.info("Warming up model...")
        num_tokens = min(
            max(16, self.max_num_reqs),
            self.scheduler_config.max_num_batched_tokens,
        )
        with _set_spyre_compilation_settings(self.vllm_config):
            self._dummy_run(num_tokens)
        logger.info("Warmup done.")

    # --- KV cache allocation ---
    # Potential sub to override KV cache tensor allocation
    # def _allocate_kv_cache_tensors(
    #     self,
    #     kv_cache_config,
    # ) -> dict[str, torch.Tensor]:

    # --- Stubs copied from CPUModelRunner ---
    # These are trivial overrides that GPUModelRunner expects.

    def _init_device_properties(self) -> None:
        # No CUDA/GPU device properties to query for Spyre
        pass

    def _sync_device(self) -> None:
        # TODO: Replace with torch.spyre.synchronize() when available.
        # For now, all copies are synchronous (no non_blocking), so
        # explicit sync is not needed.
        pass

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        return 0, None

    def get_model(self) -> nn.Module:
        # Return the unwrapped model for isinstance checks
        # (e.g. is_text_generation_model in get_supported_tasks).
        model = self.model
        if isinstance(model, _SpyreModelWrapper):
            model = model._model
        # Unwrap torch.compile's OptimizedModule (has _orig_mod attribute)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    # --- Buffer management ---

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> SpyreCpuGpuBuffer:
        """Create a SpyreCpuGpuBuffer with float tensors on Spyre.

        - Float dtypes: .cpu on CPU, .gpu on Spyre as float16
        - Int/bool dtypes: .gpu aliased to .cpu (stays on CPU)
        """
        if dtype.is_floating_point:
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=torch.float16,
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )
        # Int/bool → CPU-only (aliased)
        return SpyreCpuGpuBuffer(
            *size,
            cpu_dtype=dtype,
            gpu_dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=numpy,
        )


@contextmanager
def _set_spyre_compilation_settings(config: VllmConfig):
    """Context manager for Spyre-specific compilation settings during warmup.

    Similar to _set_global_compilation_settings in cpu_model_runner.py but
    adapted for Spyre's compilation requirements.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
