from functools import wraps
from typing import Callable

from mygrad import Tensor

from tests.utils.stateful import clear_all_mem_locking_state
from tests.utils.checkers import expected_constant


def clears_mem_state(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        finally:
            clear_all_mem_locking_state()

    return wrapper


def adds_constant_arg(func):
    """Compatibility shim so that functions that do not take a `constant`
    kwarg can be tested by the testing fixtures that require them to."""

    @wraps(func)
    def wrapper(*args, constant=None, **kwargs):
        out = func(*args, **kwargs)
        if isinstance(out, Tensor):
            out._constant = expected_constant(
                args, dest_dtype=out.dtype, constant=constant
            )
        return out

    return wrapper