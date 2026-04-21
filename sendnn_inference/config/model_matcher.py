"""Pattern-based model matching."""

from typing import Any

from vllm.logger import init_logger

from sendnn_inference.config.model_config import ArchitecturePattern

logger = init_logger(__name__)


class ModelMatcher:
    """Pattern-based model matching for identifying models from HF configs."""

    def _validate_sub_config(
        self, model_name: str, attr_name: str, config_value: Any, pattern_value: dict
    ) -> bool:
        """Validate a nested config attribute against a dict pattern.

        Handles both plain dict config values (e.g. quantization_config) and
        sub-config objects (e.g. text_config, vision_config). For dicts, uses
        dict key lookup; for objects, uses getattr.

        Args:
            model_name: Model name for logging purposes
            attr_name: Name of the parent attribute (e.g. 'quantization_config', 'text_config')
            config_value: Actual config value from HF config (dict or sub-config object)
            pattern_value: Expected pattern dict

        Returns:
            True if all pattern keys match, False otherwise
        """
        for key, value in pattern_value.items():
            if isinstance(config_value, dict):
                present = key in config_value
                actual = config_value.get(key)
            else:
                present = hasattr(config_value, key)
                actual = getattr(config_value, key) if present else None

            if not present:
                logger.debug(
                    "Model '%s': %s missing attribute '%s' required by pattern",
                    model_name,
                    attr_name,
                    key,
                )
                return False
            if actual != value:
                logger.debug(
                    "Model '%s': %s.%s mismatch: config=%s, pattern=%s",
                    model_name,
                    attr_name,
                    key,
                    actual,
                    value,
                )
                return False
        return True

    def _validate_attribute(
        self, hf_config: Any, model_name: str, attr_name: str, pattern_value: Any
    ) -> bool:
        """Validate a single attribute match between config and pattern.

        Args:
            hf_config: HuggingFace model configuration object
            model_name: Model name for logging purposes
            attr_name: Name of the attribute to validate
            pattern_value: Expected value for the attribute (never None)

        Returns:
            True if the attribute matches, False otherwise
        """
        if not hasattr(hf_config, attr_name):
            logger.debug(
                "Model '%s': HF config missing attribute '%s' required by pattern",
                model_name,
                attr_name,
            )
            return False

        config_value = getattr(hf_config, attr_name)

        if isinstance(pattern_value, dict):
            return self._validate_sub_config(model_name, attr_name, config_value, pattern_value)

        if config_value != pattern_value:
            logger.debug(
                "Model '%s': Attribute '%s' mismatch: config=%s, pattern=%s",
                model_name,
                attr_name,
                config_value,
                pattern_value,
            )
            return False

        return True

    def matches(self, hf_config: Any, pattern: ArchitecturePattern) -> bool:
        """Check if HF config matches architecture pattern.

        Args:
            hf_config: HuggingFace model configuration object
            pattern: Architecture pattern to match against

        Returns:
            True if the config matches the pattern, False otherwise
        """
        model_name = pattern.model_name

        if not hasattr(hf_config, "model_type"):
            logger.debug("Model '%s': HF config missing 'model_type' attribute", model_name)
            return False

        if hf_config.model_type != pattern.model_type:
            logger.debug(
                "Model '%s': Model type mismatch: config=%s, pattern=%s",
                model_name,
                hf_config.model_type,
                pattern.model_type,
            )
            return False

        for attr_name, pattern_value in pattern.attributes.items():
            if not self._validate_attribute(hf_config, model_name, attr_name, pattern_value):
                return False

        logger.debug("Model '%s': HF config matches pattern", model_name)
        return True


# Made with Bob
