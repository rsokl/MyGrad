from typing import Callable, Union

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import assume, given

from mygrad import Tensor


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


@pytest.mark.parametrize("func", [plus, minus, multiply, power, matmul])
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
):
    tensor = Tensor(arr)
    assert isinstance(func(tensor, arr), Tensor)
    assert isinstance(func(arr, tensor), Tensor)
