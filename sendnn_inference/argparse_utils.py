"""
Utilities for conditional argument defaults in argparse.

This module provides a mechanism to set argument defaults that depend on
the values of other arguments, which is not natively supported by argparse.

Example usage:
    from sendnn_inference.argparse_utils import ConditionalDefaultManager

    @classmethod
    def pre_register_and_update(cls, parser):
        def _compute_config_format(namespace: argparse.Namespace) -> str:
            model = getattr(namespace, 'model', '') or ''
            return 'mistral' if 'mistral' in model.lower() else 'auto'

        # Register conditional defaults that apply globally
        ConditionalDefaultManager.register(
            dest='config_format',
            compute_default=_compute_config_format,
        )

        # Apply the patches to this parser
        ConditionalDefaultManager.apply(parser)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Protocol

import argparse
import logging

from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = logging.getLogger(__name__)


class ComputeDefaultFunc(Protocol):
    """Protocol for a callable that computes a default value from a namespace."""

    def __call__(self, namespace: argparse.Namespace) -> Any: ...


class ConditionalDefaultAction(argparse.Action):
    """
    Action that marks an argument as explicitly set by the user.

    This allows us to distinguish between user-provided values and
    defaults (both static and conditional).
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        # Mark this argument as explicitly provided by the user
        explicit_attr = f"_{self.dest}_explicit"
        setattr(namespace, explicit_attr, True)
        setattr(namespace, self.dest, values)


class ConditionalDefaultManager:
    """
    Manages conditional defaults for argparse arguments.

    This class allows you to define argument defaults that depend on
    the values of other arguments, which is not natively supported by argparse.

    The mechanism works by:
    1. Replacing the standard action for each managed argument with
       ConditionalDefaultAction, which tracks if the user explicitly set it.
    2. Patching the parser's parse_args method to apply conditional defaults
       after all arguments have been parsed.

    All methods are class methods since the state is global across all parsers
    via the class variable _all_conditional_defaults.
    """

    _all_conditional_defaults: ClassVar[dict[str, ComputeDefaultFunc]] = {}

    @classmethod
    def clear(cls) -> None:
        """Clear all registered conditional defaults.

        This is useful for testing to ensure clean state between tests.
        Note that this does not unpatch ArgumentParser.parse_args.
        """
        cls._all_conditional_defaults.clear()

    @classmethod
    def register(
        cls,
        dest: str,
        compute_default: ComputeDefaultFunc,
    ) -> None:
        """
        Register a conditional default for an argument.

        Args:
            dest: The argument destination name (e.g., 'config_format').
            compute_default: A callable that takes the parsed namespace and
                             returns the default value to use. Return None to
                             skip applying a default.

        Raises:
            ValueError: If a conditional default for this dest is already registered.
        """
        if (
            dest in cls._all_conditional_defaults
            and cls._all_conditional_defaults[dest] != compute_default
        ):
            raise ValueError(
                f"Conditional default for '{dest}' is already registered. "
                f"Each destination can only be registered once."
            )
        cls._all_conditional_defaults[dest] = compute_default

    @classmethod
    def apply(cls, parser: FlexibleArgumentParser) -> None:
        """
        Apply the conditional default logic to the parser.

        This method:
        1. Replaces the action for each managed argument with ConditionalDefaultAction
        2. Patches the parser's parse_args method to apply conditional defaults
        """
        # Step 1: Replace actions for managed arguments
        for dest in cls._all_conditional_defaults:
            for action in parser._actions:
                if hasattr(action, "dest") and action.dest == dest:
                    action.__class__ = ConditionalDefaultAction
                    break

        # Step 2: Patch parse_args at the base ArgumentParser class level
        # This ensures it works even when the parser is used as a sub-parser
        cls._patch_parse_args()

    @classmethod
    def _patch_parse_args(cls) -> None:
        """Patch ArgumentParser.parse_args to apply conditional defaults."""
        import argparse as _argparse

        # Check if we've already patched the base class
        if getattr(_argparse.ArgumentParser, "_spyre_conditional_defaults_patched", False):
            logger.debug("ArgumentParser.parse_args already patched, skipping")
            return

        logger.debug(
            "Patching ArgumentParser.parse_args to apply %d conditional default(s)",
            len(cls._all_conditional_defaults),
        )

        original_parse_args = _argparse.ArgumentParser.parse_args

        def patched_parse_args(
            self: argparse.ArgumentParser,
            args: Sequence[str] | None = None,
            namespace: argparse.Namespace | None = None,
        ) -> argparse.Namespace:
            result = original_parse_args(self, args, namespace)
            assert result is not None  # type: ignore[redundant-expr]

            if args is None or len(args) == 0:
                # Don't override anything if there were no args parsed
                return result

            # Apply conditional defaults for any managed arguments
            for dest, compute_default in cls._all_conditional_defaults.items():
                # Skip if already applied or if user explicitly set the value
                applied_attr = f"_{dest}_conditional_default_applied"
                if getattr(result, applied_attr, False):
                    logger.debug(
                        "Conditional default for '%s' was already applied: skipping...",
                        dest,
                    )
                    continue
                explicit_attr = f"_{dest}_explicit"
                if getattr(result, explicit_attr, False):
                    logger.debug(
                        "Skipping conditional default for '%s': user explicitly provided value",
                        dest,
                    )
                    continue

                # Apply the conditional default
                try:
                    value = compute_default(result)
                    if value is not None:
                        logger.info(
                            "Applying conditional default for '%s': %r",
                            dest,
                            value,
                        )
                        setattr(result, dest, value)
                        setattr(result, applied_attr, True)
                except Exception as e:
                    logger.error(
                        "Failed to compute conditional default for '%s': %s",
                        dest,
                        e,
                    )

            return result

        _argparse.ArgumentParser.parse_args = patched_parse_args  # type: ignore[invalid-assignment]
        _argparse.ArgumentParser._spyre_conditional_defaults_patched = True  # type: ignore[attr-defined]
