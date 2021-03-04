from functools import partial
from typing import Callable, Type, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import Tensor
from mygrad.linalg.ops import MatMul
from mygrad.math.arithmetic.ops import (
    Add,
    Divide,
    Multiply,
    Positive,
    Power,
    Square,
    Subtract,
)
from mygrad.operation_base import Operation


def plus(x, y):
    return x + y


def minus(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    return x / y


def power(x, y):
    return x ** y


def matmul(x, y):
    assume(0 < x.ndim < 3)
    return x @ y.T


@pytest.mark.parametrize(
    "func, op",
    [
        (plus, Add),
        (minus, Subtract),
        (multiply, Multiply),
        (divide, Divide),
        (power, (Power, Positive, Square)),  # can specialize
        (matmul, MatMul),
    ],
)
@given(
    arr=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        dtype=hnp.floating_dtypes(),
        elements=dict(min_value=1.0, max_value=2.0),
    )
)
def test_arithmetic_operators_between_array_and_tensor_cast_to_tensor(
    arr: np.ndarray,
    func: Callable[[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]], Tensor],
    op: Type[Operation],
):
    tensor = Tensor(arr)
    out = func(tensor, arr)
    assert isinstance(out, Tensor)
    assert isinstance(out.creator, op)

    out = func(arr, tensor)
    assert isinstance(out, Tensor)
    assert isinstance(out.creator, op)

    out = func(tensor, tensor)
    assert isinstance(out, Tensor)
    assert isinstance(out.creator, op)


constant_tensor: Callable[..., Tensor] = partial(mg.tensor, constant=True)


@given(
    arr1=hnp.arrays(
        shape=st.just(tuple()) | st.just((3,)),
        dtype=st.sampled_from([float, int]),
        elements=dict(min_value=1, max_value=2),
    ),
    arr2=hnp.arrays(
        shape=st.just(tuple()) | st.just((3,)),
        dtype=st.sampled_from([float, int]),
        elements=dict(min_value=1, max_value=2),
    ),
)
@pytest.mark.parametrize(
    "f1, f2",
    [
        (constant_tensor, lambda x: x.tolist()),
        (lambda x: x, constant_tensor),
        (constant_tensor, constant_tensor),
    ],
)
def test_floor_div(arr1, arr2, f1, f2):
    desired = arr1 // arr2
    actual = f1(arr1) // f2(arr2)
    assert actual.dtype == desired.dtype
    assert_array_equal(desired, actual)


def test_floor_div_is_raises_for_variable_tensors():
    with pytest.raises(ValueError):
        mg.tensor(1.0, constant=False) // 1

    with pytest.raises(ValueError):
        1 // mg.tensor(1.0, constant=False)
