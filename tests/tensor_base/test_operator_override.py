from typing import Callable, Type, Union

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import assume, given

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
