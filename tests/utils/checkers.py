from typing import Optional, Union

import numpy as np

import mygrad
import mygrad as mg
from mygrad import Tensor
from mygrad.operation_base import _NoValue, _NoValueType
from mygrad.tensor_base import CONSTANT_ONLY_DTYPES
from mygrad.typing import ArrayLike, DTypeLike, DTypeLikeReals
from tests.utils.errors import InternalTestError


def check_consistent_grad_dtype(*args: ArrayLike):
    """Raises assertion error if `t.grad.dtype` does not match `t.dtype`
    for variable tensors."""
    for item in args:
        if not isinstance(item, mg.Tensor):
            continue
        elif item.constant:
            assert item.grad is None
        else:
            assert item.grad.dtype == item.dtype


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
    constant: Optional[Union[_NoValueType, bool]] = None,
) -> bool:
    """Given the input arguments to a function that produces a tensor, infers
    whether the resulting tensor should be a constant or a variable tensor."""
    if constant is _NoValue:
        constant = None

    if not isinstance(constant, bool) and constant is not None:
        raise TypeError(f"Invalid type for `constant`; got: {constant}")

    if dest_dtype is None:
        raise InternalTestError("`dest_dtype` cannot be `None`")

    dest_dtype = np.dtype(dest_dtype)
    is_constant_only_dtype = issubclass(dest_dtype.type, CONSTANT_ONLY_DTYPES)

    if constant is not None:
        if constant is False and dest_dtype is not None and is_constant_only_dtype:
            raise InternalTestError(
                f"dest_dtype ({dest_dtype}) and specified constant ({constant}) are inconsistent"
            )
        return constant

    if is_constant_only_dtype:
        return True

    elif args:
        # constant inferred from inputs
        for item in args:
            # if any input is a variable-tensor, output is variable
            if isinstance(item, Tensor) and item.constant is False:
                return False
        return True
    else:
        return False


def is_float_arr(arr: Union[np.ndarray, mygrad.Tensor]) -> bool:
    return issubclass(arr.dtype.type, np.floating)
