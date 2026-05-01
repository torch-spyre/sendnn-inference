"""Model configurator for applying model-specific configurations."""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from sendnn_inference import envs as envs_spyre

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from sendnn_inference.config.model_config import DeviceConfig, ModelConfig

logger = init_logger(__name__)


@dataclass
class ConfigValue:
    """Tracks a configuration value with override information.

    Attributes:
        default: The expected/default value from model configuration
        applied: The actual value that was applied (may differ if overridden)
    """

    default: str | int | None
    applied: str | int | None

    def was_overridden(self) -> bool:
        """Check if the applied value differs from the default value."""
        return self.default != self.applied

    def __eq__(self, other: object) -> bool:
        """Compare ConfigValue with another value using the applied value.

        This allows code that compares directly to values
        """
        return self.applied == other


@dataclass
class ConfigurationSummary:
    """Summary of configuration changes applied by the configurator.

    Attributes:
        model_name: Name of the model being configured
        tp_size: Tensor parallel size
        env_vars: Dictionary of environment variables with override tracking
        num_blocks: num_gpu_blocks_override value with override tracking, if configured
    """

    model_name: str
    tp_size: int
    env_vars: dict[str, ConfigValue] = field(default_factory=dict)
    num_blocks: ConfigValue | None = None

    def format_log_message(self) -> str:
        """Format the configuration summary as a multi-line log message.

        Returns:
            Formatted string suitable for logging with logger.info()
        """

        def format_config_line(name: str, config_value: ConfigValue) -> str:
            if config_value.was_overridden():
                return f"  {name}={config_value.applied} ⚠ (default: {config_value.default})"
            return f"  {name}={config_value.applied} ✓"

        def generate_lines():
            yield f"Applied registry configuration for '{self.model_name}' (TP={self.tp_size}):"

            if self.env_vars:
                yield "  Environment variables:"
                for key, config_value in self.env_vars.items():
                    yield f"  {format_config_line(key, config_value)}"

            if self.num_blocks is not None:
                yield format_config_line("num_gpu_blocks_override", self.num_blocks)

        lines = list(generate_lines())
        if len(lines) == 1:
            lines.append("  no device-specific configs")

        return "\n".join(lines)


class ModelConfigurator:
    """Configurator that handles all model configurations.

    This configurator applies device configurations including:
    - Environment variables
    - GPU block overrides (with version-aware logic)

    All features are optional and driven by the device_config in YAML.
    """

    def __init__(self, model_config: "ModelConfig", device_config: "DeviceConfig | None" = None):
        """Initialize configurator with model configuration and optional device config.

        Args:
            model_config: The model configuration to use
            device_config: Optional device configuration (from matching CB config)
        """
        self.model_config = model_config
        self.device_config = device_config

    def configure(self, vllm_config: "VllmConfig") -> ConfigurationSummary:
        """Apply device configurations.

        Args:
            vllm_config: The vLLM configuration to modify

        Returns:
            ConfigurationSummary with all configuration settings checked/applied
        """
        tp_size = vllm_config.parallel_config.world_size

        summary = ConfigurationSummary(
            model_name=self.model_config.name,
            tp_size=tp_size,
        )

        if self.device_config is None:
            logger.debug(
                "No device configuration for model '%s' with TP=%d",
                self.model_config.name,
                tp_size,
            )
            return summary

        # Apply environment variables and track them
        for key, value in self.device_config.env_vars.items():
            config_value = self.set_env_var(key, value, override=False)
            summary.env_vars[key] = config_value

        # Handle num_gpu_blocks_override with version check
        blocks_override = self._configure_gpu_blocks(self.device_config, vllm_config)
        if blocks_override is not None:
            summary.num_blocks = blocks_override

        return summary

    def _validate_config_override(
        self,
        config_name: str,
        config_value: ConfigValue,
        error_context: str,
    ) -> None:
        """Validate that applied config value matches default value.

        Handles the common pattern of checking if a user-provided value conflicts
        with the default model configuration value, and enforces
        SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG.

        Args:
            config_name: Name of the configuration parameter (for error messages)
            config_value: ConfigValue with default and applied values
            error_context: Additional context for error messages (e.g., "it was already set to X")

        Raises:
            RuntimeError: If SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG is set and values conflict
        """
        if config_value.was_overridden():
            if envs_spyre.SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG:
                raise RuntimeError(
                    f"Model '{self.model_config.name}' configures "
                    f"{config_name}={config_value.default}, "
                    f"but {error_context}. "
                    f"SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG is enabled."
                )
            logger.warning(
                "%s was set to %s, not using model default of %s",
                config_name,
                config_value.applied,
                config_value.default,
            )

    def set_env_var(
        self, key: str, value: Any, override: bool = False, log_level: str = "debug"
    ) -> ConfigValue:
        """Set environment variable with logging.

        Args:
            key: Environment variable name
            value: Value to set
            override: Whether to override existing value
            log_level: Logging level ('info', 'warning', 'debug')

        Returns:
            ConfigValue tracking default and applied values

        Raises:
            RuntimeError: If SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG is set and existing value
            conflicts
        """
        str_value = str(value)
        existing = os.getenv(key)

        if existing is not None and not override:
            config_value = ConfigValue(default=str_value, applied=existing)
            if config_value.was_overridden():
                self._validate_config_override(
                    config_name=key,
                    config_value=config_value,
                    error_context=f"it was already set to {existing}",
                )
            return config_value

        os.environ[key] = str_value
        log_func = getattr(logger, log_level)
        log_func("Set %s = %s", key, str_value)
        return ConfigValue(default=str_value, applied=str_value)

    def _configure_gpu_blocks(self, device_config, vllm_config: "VllmConfig") -> ConfigValue | None:
        """Configure GPU blocks override.

        Args:
            device_config: Device configuration containing block override settings
            vllm_config: The vLLM configuration to modify

        Returns:
            ConfigValue tracking default and applied num_blocks, or None if not configured

        Raises:
            RuntimeError: If SENDNN_INFERENCE_REQUIRE_KNOWN_CONFIG is set and user override
            conflicts
        """
        num_blocks_override = device_config.num_gpu_blocks_override
        if num_blocks_override is None:
            return None

        # Apply override if not already set
        if vllm_config.cache_config.num_gpu_blocks_override is None:
            vllm_config.cache_config.num_gpu_blocks_override = num_blocks_override
            logger.debug(
                "Set num_gpu_blocks_override=%d for model %s",
                num_blocks_override,
                self.model_config.name,
            )
            return ConfigValue(default=num_blocks_override, applied=num_blocks_override)

        # User already set a value - validate it
        user_value = vllm_config.cache_config.num_gpu_blocks_override
        config_value = ConfigValue(default=num_blocks_override, applied=user_value)
        self._validate_config_override(
            config_name="num_gpu_blocks_override",
            config_value=config_value,
            error_context=f"user set --num-gpu-blocks-override={user_value}",
        )
        return config_value


# Made with Bob
