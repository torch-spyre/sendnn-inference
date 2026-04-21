"""Model configuration registry."""

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from vllm.logger import init_logger

from sendnn_inference import envs
from sendnn_inference.config.model_config import ModelConfig
from sendnn_inference.config.model_matcher import ModelMatcher

if TYPE_CHECKING:
    from vllm.config import ModelConfig as VllmModelConfig, VllmConfig

    from sendnn_inference.config.configurators.model_configurator import ModelConfigurator
    from sendnn_inference.config.model_config import DeviceConfig, WarmupShape

logger = init_logger(__name__)


class ModelConfigRegistry:
    """Singleton registry for model configurations.

    This registry manages model configurations loaded from YAML files
    and provides methods to match models and retrieve configurators.
    """

    _instance: "ModelConfigRegistry | None" = None
    _initialized: bool = False

    def __init__(self):
        """Initialize the registry."""
        self._models: dict[str, ModelConfig] = {}
        self._configurators: dict[str, ModelConfigurator] = {}
        self._matcher = ModelMatcher()

    @classmethod
    def get_instance(cls) -> "ModelConfigRegistry":
        """Get singleton instance.

        Returns:
            The singleton ModelConfigRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, config_path: Path | None = None) -> None:
        """Load configurations from YAML file.

        Args:
            config_path: Path to model_configs.yaml file. If None, uses default location.

        Note:
            Registry validation is only performed when running on Spyre device (sendnn backend).
        """
        if self._initialized:
            logger.debug("Registry already initialized, skipping")
            return

        resolved_path = self._resolve_config_path(config_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model configuration file not found at {config_path}")

        self._load_and_register_models(resolved_path)
        self._initialized = True

    def _resolve_config_path(self, config_path: Path | None) -> Path:
        """Resolve config path to absolute path.

        Priority order:
        1. Explicit config_path parameter
        2. SENDNN_INFERENCE_MODEL_CONFIG_FILE environment variable
        3. Default location (sendnn_inference/config/model_configs.yaml)
        """
        if config_path is not None:
            return config_path

        env_path = envs.SENDNN_INFERENCE_MODEL_CONFIG_FILE
        if env_path is not None:
            return Path(env_path)

        return Path(__file__).parent / "model_configs.yaml"

    def _load_and_register_models(self, config_path: Path) -> None:
        """Load YAML and register all models.

        Args:
            config_path: Path to the configuration file

        Raises:
            RuntimeError: If loading or parsing fails
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError as e:
            # This should never happen since _validate_config_path checks file existence
            raise AssertionError(f"Unreachable: File validated but not found: {e}") from e
        except yaml.YAMLError as e:
            logger.error("YAML parsing error: %s", e)
            raise RuntimeError(f"Failed to load model configurations: {e}") from e
        except OSError as e:
            logger.error("File read error: %s", e)
            raise RuntimeError(f"Failed to load model configurations: {e}") from e

        # Handle empty YAML files (yaml.safe_load returns None)
        if data is None:
            data = {}

        for model_name, model_data in data.get("models", {}).items():
            model_config = ModelConfig.from_dict(model_name, model_data)
            self.register_model(model_config)

        logger.info("Loaded %d model configurations from %s", len(self._models), config_path)

    def register_model(self, model_config: ModelConfig) -> None:
        """Register a model configuration.

        Args:
            model_config: The model configuration to register
        """
        if model_config.name in self._models:
            logger.warning(
                "Model '%s' is already registered and will be overwritten", model_config.name
            )
        self._models[model_config.name] = model_config
        logger.debug("Registered model: %s", model_config.name)

    def find_matching_model(self, vllm_model_config: "VllmModelConfig") -> ModelConfig | None:
        """Find model config by matching HF config.

        When multiple patterns match, returns the one with the most fields
        to prioritize more specific patterns (e.g., quantized models) over generic ones.

        Args:
            vllm_model_config: vLLM model configuration containing HF config

        Returns:
            ModelConfig if a match is found, None otherwise
        """
        hf_config = vllm_model_config.hf_config
        if hf_config is None:
            logger.debug("No HF config available for matching")
            return None

        best_match: ModelConfig | None = None
        best_field_count = -1

        for model_name, model_config in self._models.items():
            if self._matcher.matches(hf_config, model_config.architecture):
                field_count = model_config.architecture.field_count
                logger.debug(
                    "Model '%s' matches configuration '%s' with %d fields",
                    vllm_model_config.model,
                    model_name,
                    field_count,
                )

                if field_count > best_field_count:
                    best_match = model_config
                    best_field_count = field_count

        if best_match:
            logger.info(
                "Matched model '%s' to configuration '%s' (%d fields)",
                vllm_model_config.model,
                best_match.name,
                best_field_count,
            )
        else:
            logger.debug("No matching model configuration found for '%s'", vllm_model_config.model)

        return best_match

    def get_configurator_for_runtime(
        self,
        vllm_config: "VllmConfig",
        warmup_shapes: list[tuple[int, int]] | None = None,
    ) -> "ModelConfigurator | None":
        """Get configurator for a model with runtime-specific device config.

        This method:
        1. Finds the matching model by HF config
        2. Verifies there's a supported runtime configuration
        3. Finds the appropriate device_config (if any) for the runtime parameters
        4. Returns a configurator (possibly with no device_config if none matches)

        A registry match requires BOTH:
        - Model architecture pattern match
        - Runtime config (static or continuous batching) matching the runtime params

        Args:
            vllm_config: vLLM configuration containing model and runtime parameters
            warmup_shapes: Optional warmup shapes for static batching validation
                (prompt_len, batch_size) tuples for pooling models

        Returns:
            ModelConfigurator instance if model matches AND has supported runtime config,
            None if no model match OR no supported runtime config
        """
        model_config = self.find_matching_model(vllm_config.model_config)
        if model_config is None:
            logger.debug("No model architecture match found")
            return None

        has_runtime_match, device_config = self._find_runtime_match_and_device_config(
            model_config, vllm_config, warmup_shapes
        )

        if not has_runtime_match:
            logger.warning(
                "Model '%s' registered but does not support the requested runtime configuration",
                model_config.name,
            )
            return None

        return self._create_configurator(model_config, device_config)

    def _find_runtime_match_and_device_config(
        self,
        model_config: ModelConfig,
        vllm_config: "VllmConfig",
        warmup_shapes: list[tuple[int, int]] | None = None,
    ) -> tuple[bool, "DeviceConfig | None"]:
        """Find matching runtime config and associated device config.

        This method searches for a runtime configuration that matches the current
        vLLM configuration. For continuous batching configs, it also extracts the
        associated device_config if present.

        Args:
            model_config: Model configuration
            vllm_config: vLLM configuration
            warmup_shapes: Optional warmup shapes for static batching validation
                (prompt_len, batch_size) tuples for pooling models

        Returns:
            Tuple of (has_match, device_config) where:
            - has_match: True if a supported runtime config exists
            - device_config: Associated device config, or None
        """
        tp_size = vllm_config.parallel_config.world_size
        max_model_len = vllm_config.model_config.max_model_len
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        if not warmup_shapes:
            cb_result = self._match_cb_config(model_config, tp_size, max_model_len, max_num_seqs)
            if cb_result is not None:
                return cb_result
            return False, None

        if self._has_static_batching_match(model_config, tp_size, warmup_shapes):
            logger.debug(
                "Found static batching config for model '%s' with TP=%d"
                "and compatible warmup shapes.",
                model_config.name,
                tp_size,
            )
            return True, None

        return False, None

    def _has_static_batching_match(
        self,
        model_config: ModelConfig,
        tp_size: int,
        warmup_shapes: list[tuple[int, int]],
    ) -> bool:
        """Check if static batching config matches runtime parameters.

        Args:
            model_config: Model configuration
            tp_size: Tensor parallel size
            warmup_shapes: Warmup shapes from runtime (prompt_len, batch_size) tuples

        Returns:
            True if a matching static batching config exists
        """
        return any(
            sb_config.tp_size == tp_size
            and self._warmup_shapes_compatible(sb_config.warmup_shapes, warmup_shapes)
            for sb_config in model_config.static_batching_configs
        )

    def _match_cb_config(
        self,
        model_config: ModelConfig,
        tp_size: int,
        max_model_len: int,
        max_num_seqs: int,
    ) -> tuple[bool, "DeviceConfig | None"] | None:
        """Match continuous batching configuration.

        Args:
            model_config: Model configuration
            tp_size: Tensor parallel size
            max_model_len: Maximum model length
            max_num_seqs: Maximum number of sequences

        Returns:
            Tuple of (has_match, device_config) if match found, None otherwise
        """
        for cb_config in model_config.continuous_batching_configs:
            if (
                cb_config.tp_size != tp_size
                or cb_config.max_model_len != max_model_len
                or cb_config.max_num_seqs != max_num_seqs
            ):
                continue

            logger.debug(
                "Found continuous batching config match for model '%s' "
                "(TP=%d, max_model_len=%d, max_num_seqs=%d)",
                model_config.name,
                tp_size,
                max_model_len,
                max_num_seqs,
            )
            return True, cb_config.device_config

        return None

    def _warmup_shapes_compatible(
        self, config_shapes: list["WarmupShape"], runtime_shapes: list[tuple[int, int]]
    ) -> bool:
        """Check if runtime warmup shapes are compatible with config warmup shapes.

        Runtime shapes are compatible if they are a subset of config shapes.

        Args:
            config_shapes: WarmupShapes from model config
            runtime_shapes: Warmup shapes from runtime
                [(prompt_len, batch_size), ...]

        Returns:
            True if all runtime shapes are in config shapes
        """
        if not runtime_shapes:
            return False

        # Runtime shapes must be a subset of config shapes
        config_set = {shape.to_tuple() for shape in config_shapes}
        runtime_set = set(runtime_shapes)
        return runtime_set.issubset(config_set)

    def _create_configurator(
        self, model_config: ModelConfig, device_config: "DeviceConfig | None" = None
    ) -> "ModelConfigurator":
        """Create configurator instance.

        Args:
            model_config: Model configuration
            device_config: Optional device configuration

        Returns:
            ModelConfigurator instance
        """
        from sendnn_inference.config.configurators.model_configurator import ModelConfigurator

        logger.debug("Creating configurator for model %s", model_config.name)
        return ModelConfigurator(model_config, device_config)

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns:
            List of model names
        """
        return list(self._models.keys())


def get_model_registry() -> ModelConfigRegistry:
    """Get the global model registry instance.

    This is a convenience function that ensures the registry is initialized.

    Returns:
        The initialized ModelConfigRegistry instance
    """
    registry = ModelConfigRegistry.get_instance()
    if not registry._initialized:
        registry.initialize()
    return registry


# Made with Bob
