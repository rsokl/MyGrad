from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Optional, Union

import numpy as np

import mygrad._utils.lock_management as mem
from mygrad import Tensor
from mygrad.typing import ArrayLike, DTypeLike, DTypeLikeReals


class InternalTestError(Exception):
    """Marks errors that are caused by bad test configurations"""


def check_dtype_consistency(out: ArrayLike, dest_dtype: DTypeLike):
    if dest_dtype is None:
        return

    out = np.asarray(out)
    dest_dtype = np.dtype(dest_dtype)
    assert (
        out.dtype == dest_dtype
    ), f"Mismatched dtypes.\nSpecified dtype: {dest_dtype}\ndtype of output: {out.dtype}"


def expected_constant(
    *args: ArrayLike,
    dest_dtype: DTypeLikeReals,
    constant: Optional[bool] = None,
) -> bool:
    """Given the input arguments to a function that produces a tensor, infers
    whether the resulting tensor should be a constant or a variable tensor."""
    if not isinstance(constant, bool) and constant is not None:
        raise TypeError(f"Invalid type for `constant`; got: {constant}")

    if dest_dtype is None:
        raise InternalTestError("`dest_dtype` cannot be `None`")

    dest_dtype = np.dtype(dest_dtype)
    is_integer_dtype = issubclass(dest_dtype.type, np.integer)

    if constant is not None:
        if constant is False and dest_dtype is not None and is_integer_dtype:
            raise InternalTestError(
                f"dest_dtype ({dest_dtype}) and specified constant ({constant}) are inconsistent"
            )
        return constant

    if issubclass(dest_dtype.type, np.integer):
        return True

    if args:
        # constant inferred from inputs
        for item in args:
            # if any input is a variable-tensor, output is variable
            if isinstance(item, Tensor) and item.constant is False:
                return False
        return True

    # constant is inferred from the dtype
    return issubclass(dest_dtype.type, np.integer)


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
