"""
Utilities for conditional argument defaults in argparse.

This module provides a mechanism to set argument defaults that depend on
the values of other arguments, which is not natively supported by argparse.

Example usage:
    from vllm_spyre.argparse_utils import ConditionalDefaultManager, register_conditional_default

    @classmethod
    def pre_register_and_update(cls, parser):
        # Register conditional defaults that apply globally
        register_conditional_default(
            dest='config_format',
            compute_default=lambda args: 'mistral' if 'mistral' args.model.lower() else 'auto',
        )

        manager = ConditionalDefaultManager(parser)
        manager.apply()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

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
    """

    def __init__(self, parser: FlexibleArgumentParser) -> None:
        self.parser = parser

    def apply(self) -> None:
        """
        Apply the conditional default logic to the parser.

        This method:
        1. Replaces the action for each managed argument with ConditionalDefaultAction
        2. Patches the parser's parse_args method to apply conditional defaults
        """
        logger.debug(
            "Enabling conditional defaults with %d config(s)",
            len(_all_conditional_defaults),
        )

        # Step 1: Replace actions for managed arguments
        seen_dests: set[str] = set()
        for config in _all_conditional_defaults:
            dest = config["dest"]
            if dest in seen_dests:
                continue
            seen_dests.add(dest)
            for action in self.parser._actions:
                if hasattr(action, "dest") and action.dest == dest:
                    action.__class__ = ConditionalDefaultAction
                    break

        # Step 2: Patch parse_args at the base ArgumentParser class level
        # This ensures it works even when the parser is used as a sub-parser
        self._patch_parse_args()

    def _patch_parse_args(self) -> None:
        """Patch ArgumentParser.parse_args to apply conditional defaults."""
        import argparse as _argparse

        # Check if we've already patched the base class
        if getattr(_argparse.ArgumentParser, "_spyre_conditional_defaults_patched", False):
            logger.debug("ArgumentParser.parse_args already patched, skipping")
            return

        logger.debug(
            "Patching ArgumentParser.parse_args to apply %d conditional default(s)",
            len(_all_conditional_defaults),
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
            for config in _all_conditional_defaults:
                dest = config["dest"]

                # Skip if already applied or if user explicitly set the value
                applied_attr = f"_{dest}_conditional_default_applied"
                if getattr(result, applied_attr, False):
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
                    value = config["compute_default"](result)
                    if value is not None:
                        logger.info(
                            "Applying conditional default for '%s': %r",
                            dest,
                            value,
                        )
                        setattr(result, dest, value)
                        setattr(result, applied_attr, True)
                except Exception as e:
                    logger.debug(
                        "Failed to compute conditional default for '%s': %s",
                        dest,
                        e,
                    )

            return result

        _argparse.ArgumentParser.parse_args = patched_parse_args  # type: ignore[invalid-assignment]
        _argparse.ArgumentParser._spyre_conditional_defaults_patched = True  # type: ignore[attr-defined]


# Global registry for conditional defaults across all parsers
_all_conditional_defaults: list[dict[str, Any]] = []


def register_conditional_default(
    dest: str,
    compute_default: ComputeDefaultFunc,
) -> None:
    """
    Register a conditional default that will be applied to any parser.

    This is useful when you want to apply the same conditional default
    across multiple parsers or when you don't have direct access to the
    parser instance.

    Args:
        dest: The argument destination name.
        compute_default: A callable that takes the parsed namespace and
                         returns the default value to use. Return None to
                         skip applying a default.
    """
    _all_conditional_defaults.append(
        {
            "dest": dest,
            "compute_default": compute_default,
        }
    )
