from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Union

import numpy as np

import mygrad._utils.lock_management as mem
from mygrad import Tensor


@contextmanager
def does_not_raise():
    """An 'empty' constext manager that yields ``None``. This is
    to be used in conjunction with ``pytest.raises`` in scenarios
    where the tested function ought not raise any exception.

    Examples
    --------
    >>> import pytest
    >>> x = "hello"
    >>> with (pytest.raises(AttributeError) if not isinstance(x, str) else does_not_raise()):
        ... x.lower()
    """
    yield


array_flag_fields = (
    "ALIGNED",
    "BEHAVED",
    "C_CONTIGUOUS",
    "CARRAY",
    "CONTIGUOUS",
    "F_CONTIGUOUS",
    "FARRAY",
    "FNC",
    "FORC",
    "FORTRAN",
    "OWNDATA",
    "WRITEABLE",
    "WRITEBACKIFCOPY",
)


def flags_to_dict(x: Union[Tensor, np.ndarray]) -> Dict[str, bool]:
    arr = np.asarray(x)
    return {k: arr.flags[k] for k in array_flag_fields}


def clear_all_mem_locking_state():
    mem._views_waiting_for_unlock.clear()
    mem._array_tracker.clear()
    mem._array_counter.clear()


def clears_mem_state(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        finally:
            clear_all_mem_locking_state()

    return wrapper
