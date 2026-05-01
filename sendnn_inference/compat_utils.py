import inspect
from dataclasses import fields
from functools import lru_cache
from typing import Callable


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]


@lru_cache
def has_argument(func: Callable, param_name: str) -> bool:
    # Checks the signature of a method and returns true iff the method accepts
    # a parameter named `$param_name`.
    # `lru_cache` is used because inspect + for looping is pretty slow. This
    # should not be invoked in the critical path.
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if (
            param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.name == param_name
        ):
            return True
    return False
