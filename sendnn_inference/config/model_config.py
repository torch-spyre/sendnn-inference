"""Data structures for model configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArchitecturePattern:
    """Pattern for matching model architectures.

    Attributes:
        model_name: The model name/identifier (e.g., 'ibm-granite/granite-3.3-8b-instruct')
        model_type: The model type (e.g., 'granite', 'llama', 'roberta')
        attributes: Dictionary of attributes that MUST match against HF config.
                   Only include attributes that are required for matching.
                   Values cannot be None - only include fields needed for matching.
                   Common attributes include: num_hidden_layers, max_position_embeddings,
                   hidden_size, vocab_size, num_key_value_heads, num_attention_heads,
                   num_experts_per_tok, quantization_config, etc.
    """

    model_name: str
    model_type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def field_count(self) -> int:
        """Count the number of fields in the pattern.

        Used for pattern prioritization: patterns with more fields are more specific
        and should be matched first (e.g., quantized models have more fields than base models).

        For nested dictionaries (like quantization_config), counts the dict itself as 1
        plus each key within it as an additional 1.

        Returns:
            Number of fields in the pattern
        """
        count = len(self.attributes)

        # Add nested dict fields
        for attr_value in self.attributes.values():
            if isinstance(attr_value, dict):
                count += len(attr_value)

        return count

    @classmethod
    def from_dict(cls, model_name: str, data: dict[str, Any]) -> "ArchitecturePattern":
        """Create ArchitecturePattern from dictionary.

        Args:
            model_name: The model name/identifier
            data: Dictionary containing architecture pattern data

        Returns:
            ArchitecturePattern instance

        Raises:
            ValueError: If any attribute value is None
        """
        # Extract model_type (required)
        model_type = data["model_type"]

        # All other keys become attributes
        attributes = {k: v for k, v in data.items() if k != "model_type"}

        # Validate no None values in attributes
        for attr_name, attr_value in attributes.items():
            if attr_value is None:
                raise ValueError(
                    f"Model '{model_name}': Attribute '{attr_name}' cannot be None. "
                    f"Only include attributes that are required for matching."
                )
            # Also check nested dicts for None values
            if isinstance(attr_value, dict):
                for nested_key, nested_value in attr_value.items():
                    if nested_value is None:
                        raise ValueError(
                            f"Model '{model_name}': Nested attribute '{attr_name}.{nested_key}' "
                            f"cannot be None. Only include attributes that are required for "
                            f"matching."
                        )

        return cls(
            model_name=model_name,
            model_type=model_type,
            attributes=attributes,
        )


@dataclass
class DeviceConfig:
    """Device-specific configuration for a model.

    Attributes:
        tp_size: Tensor parallel size this config applies to
        env_vars: Environment variables to set
        num_gpu_blocks_override: Override for GPU blocks
    """

    tp_size: int
    env_vars: dict[str, Any] = field(default_factory=dict)
    num_gpu_blocks_override: int | None = None

    @classmethod
    def from_dict(cls, tp_size: int, data: dict[str, Any]) -> "DeviceConfig":
        """Create DeviceConfig from dictionary."""
        return cls(
            tp_size=tp_size,
            env_vars=data.get("env_vars", {}),
            num_gpu_blocks_override=data.get("num_gpu_blocks_override"),
        )


@dataclass
class WarmupShape:
    """Warmup shape configuration for static batching (pooling models only).

    Attributes:
        prompt_len: Prompt length
        batch_size: Batch size
    """

    prompt_len: int
    batch_size: int

    def to_tuple(self) -> tuple[int, int]:
        """Convert WarmupShape to tuple format.

        Returns:
            Tuple of (prompt_len, batch_size)
        """
        return (self.prompt_len, self.batch_size)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WarmupShape":
        """Create WarmupShape from dictionary.

        Args:
            data: Dictionary containing warmup shape data with keys:
                  prompt_len, batch_size

        Returns:
            WarmupShape instance

        Raises:
            KeyError: If required keys are missing
            ValueError: If values are not valid integers
        """
        try:
            return cls(
                prompt_len=int(data["prompt_len"]),
                batch_size=int(data["batch_size"]),
            )
        except KeyError as e:
            raise ValueError(
                f"Warmup shape must have 'prompt_len' and 'batch_size' keys. Missing key: {e}"
            ) from e
        except (ValueError, TypeError) as e:
            raise ValueError(f"Warmup shape values must be valid integers: {e}") from e


@dataclass
class StaticBatchingConfig:
    """Static batching configuration (pooling models only).

    Attributes:
        tp_size: Tensor parallel size
        warmup_shapes: List of warmup shape configurations
    """

    tp_size: int
    warmup_shapes: list[WarmupShape]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticBatchingConfig":
        """Create StaticBatchingConfig from dictionary."""
        # Convert warmup shapes from dicts to WarmupShape objects
        warmup_shapes = [WarmupShape.from_dict(ws) for ws in data["warmup_shapes"]]
        return cls(
            tp_size=data["tp_size"],
            warmup_shapes=warmup_shapes,
        )


@dataclass
class ContinuousBatchingConfig:
    """Continuous batching configuration with optional device config.

    Attributes:
        tp_size: Tensor parallel size
        max_model_len: Maximum model length
        max_num_seqs: Maximum number of sequences
        device_config: Optional device-specific configuration (nested)
    """

    tp_size: int
    max_model_len: int
    max_num_seqs: int
    device_config: "DeviceConfig | None" = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuousBatchingConfig":
        """Create ContinuousBatchingConfig from dictionary.

        The device_config field is optional and will be None if not present.
        """
        device_config = None
        if "device_config" in data:
            # tp_size is inherited from parent config
            device_config = DeviceConfig.from_dict(
                tp_size=data["tp_size"], data=data["device_config"]
            )

        return cls(
            tp_size=data["tp_size"],
            max_model_len=data["max_model_len"],
            max_num_seqs=data["max_num_seqs"],
            device_config=device_config,
        )


# Type alias for runtime configs
RuntimeConfig = StaticBatchingConfig | ContinuousBatchingConfig


@dataclass
class ModelConfig:
    """Complete model configuration.

    Attributes:
        name: Model name/identifier
        architecture: Architecture pattern for matching
        static_batching_configs: List of static batching configurations (pooling models only)
        continuous_batching_configs: List of continuous batching configurations (decoder models)
            (each may have its own device_config)
    """

    name: str
    architecture: ArchitecturePattern
    static_batching_configs: list[StaticBatchingConfig] = field(default_factory=list)
    continuous_batching_configs: list[ContinuousBatchingConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary (typically from YAML).

        Args:
            name: Model name
            data: Dictionary containing model configuration

        Returns:
            ModelConfig instance
        """
        # Parse architecture (pass model name for better logging)
        architecture = ArchitecturePattern.from_dict(name, data["architecture"])

        # Parse static batching configs
        static_configs = []
        for cfg in data.get("static_batching_configs", []):
            static_configs.append(StaticBatchingConfig.from_dict(cfg))

        # Parse continuous batching configs (with nested device configs)
        continuous_configs = []
        for cfg in data.get("continuous_batching_configs", []):
            continuous_configs.append(ContinuousBatchingConfig.from_dict(cfg))

        # Validate no duplicate CB configs
        cb_signatures = set()
        for cfg in continuous_configs:
            signature = (cfg.tp_size, cfg.max_model_len, cfg.max_num_seqs)
            if signature in cb_signatures:
                raise ValueError(
                    f"Duplicate runtime configuration for model '{name}': "
                    f"tp_size={cfg.tp_size}, max_model_len={cfg.max_model_len}, "
                    f"max_num_seqs={cfg.max_num_seqs}"
                )
            cb_signatures.add(signature)

        # Validate no duplicate static configs
        static_signatures = set()
        for cfg in static_configs:
            # Convert WarmupShape objects to tuples and sort for comparison (order shouldn't matter)
            warmup_tuples = [shape.to_tuple() for shape in cfg.warmup_shapes]
            signature = (cfg.tp_size, tuple(sorted(warmup_tuples)))
            if signature in static_signatures:
                raise ValueError(
                    f"Duplicate runtime configuration for model '{name}': "
                    f"tp_size={cfg.tp_size}, warmup_shapes={warmup_tuples}"
                )
            static_signatures.add(signature)

        # Validate at least one runtime configuration exists
        if not static_configs and not continuous_configs:
            raise ValueError(
                f"Model '{name}' must have at least one runtime configuration "
                f"(either static_batching_configs or continuous_batching_configs)"
            )

        return cls(
            name=name,
            architecture=architecture,
            static_batching_configs=static_configs,
            continuous_batching_configs=continuous_configs,
        )


# Made with Bob
