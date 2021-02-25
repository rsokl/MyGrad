from typing import NamedTuple, Tuple, Any, Dict

from mygrad.operation_base import _NoValue


class MinimalArgs(NamedTuple):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


def populate_args(*args, **kwargs) -> MinimalArgs:
    return MinimalArgs(
        args=tuple(item for item in args if item is not _NoValue),
        kwargs={k: v for k, v in kwargs.items() if v is not _NoValue},
    )