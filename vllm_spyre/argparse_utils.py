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

from vllm.utils.argparse_utils import FlexibleArgumentParser


class ComputeDefaultFunc(Protocol):
    """Protocol for a callable that computes a default value from a namespace."""

    def __call__(self, namespace: argparse.Namespace) -> Any: ...


# Track which parsers have been patched to avoid duplicate work
_PATCHED_PARSERS: set[int] = set()


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
        self._conditional_defaults: list[dict[str, Any]] = []

    def add_conditional_default(
        self,
        dest: str,
        compute_default: ComputeDefaultFunc,
        explicit_marker_attr: str | None = None,
    ) -> None:
        """
        Register a conditional default for an argument.

        Args:
            dest: The argument destination name (e.g., 'config_format').
            compute_default: A callable that takes the parsed namespace and
                             returns the default value to use. Return None to
                             skip applying a default.
            explicit_marker_attr: Optional custom attribute name for tracking
                                  whether the user explicitly set this argument.
                                  Defaults to f"_{dest}_explicit".
        """
        self._conditional_defaults.append(
            {
                "dest": dest,
                "compute_default": compute_default,
                "explicit_marker_attr": explicit_marker_attr or f"_{dest}_explicit",
            }
        )

    def apply(self) -> None:
        """
        Apply the conditional default logic to the parser.

        This method:
        1. Replaces the action for each managed argument with ConditionalDefaultAction
        2. Patches the parser's parse_args method to apply conditional defaults
        """
        # Avoid patching the same parser instance twice
        parser_id = id(self.parser)
        if parser_id in _PATCHED_PARSERS:
            return
        _PATCHED_PARSERS.add(parser_id)

        # Step 1: Replace actions for managed arguments (both local and global)
        all_configs = self._conditional_defaults + _all_conditional_defaults
        seen_dests: set[str] = set()
        for config in all_configs:
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
            return

        original_parse_args = _argparse.ArgumentParser.parse_args

        def patched_parse_args(
            self: argparse.ArgumentParser,
            args: Sequence[str] | None = None,
            namespace: argparse.Namespace | None = None,
        ) -> argparse.Namespace:
            result = original_parse_args(self, args, namespace)
            assert result is not None  # type: ignore[redundant-expr]

            # Apply conditional defaults for any managed arguments
            for config in _all_conditional_defaults:
                explicit_marker = config["explicit_marker_attr"]
                dest = config["dest"]

                # Skip if already applied or if user explicitly set the value
                applied_attr = f"_{dest}_conditional_default_applied"
                if getattr(result, applied_attr, False):
                    continue
                if getattr(result, explicit_marker, False):
                    continue

                # Apply the conditional default
                try:
                    value = config["compute_default"](result)
                    if value is not None:
                        setattr(result, dest, value)
                        setattr(result, applied_attr, True)
                except Exception:
                    # If condition evaluation fails, skip this default
                    pass

            return result

        _argparse.ArgumentParser.parse_args = patched_parse_args  # type: ignore[invalid-assignment]
        _argparse.ArgumentParser._spyre_conditional_defaults_patched = True  # type: ignore[attr-defined]


# Global registry for conditional defaults across all parsers
_all_conditional_defaults: list[dict[str, Any]] = []


def register_conditional_default(
    dest: str,
    compute_default: ComputeDefaultFunc,
    explicit_marker_attr: str | None = None,
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
        explicit_marker_attr: Optional custom attribute for tracking explicit values.
    """
    _all_conditional_defaults.append(
        {
            "dest": dest,
            "compute_default": compute_default,
            "explicit_marker_attr": explicit_marker_attr or f"_{dest}_explicit",
        }
    )
