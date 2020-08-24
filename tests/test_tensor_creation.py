import hypothesis.strategies as st
import numpy as np
from tests.custom_strategies import tensors
import hypothesis.extra.numpy as hnp
from hypothesis import given
from numpy.testing import assert_array_equal

from mygrad import Tensor, astensor
from mygrad.tensor_creation.funcs import (
    arange,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    geomspace,
    identity,
    linspace,
    logspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)


def check_tensor_array(tensor, array, constant):
    assert isinstance(tensor, Tensor)
    assert_array_equal(tensor.data, array)
    assert tensor.dtype is array.dtype
    assert tensor.constant is constant


@given(constant=st.booleans(), dtype=st.sampled_from((np.int32, np.float64)))
def test_all_tensor_creation(constant, dtype):
    x = np.array([1, 2, 3])

    e = empty((3, 2), dtype=dtype, constant=constant)
    assert e.shape == (3, 2)
    assert e.constant is constant

    e = empty_like(e, dtype=dtype, constant=constant)
    assert e.shape == (3, 2)
    assert e.constant is constant

    check_tensor_array(
        eye(3, dtype=dtype, constant=constant), np.eye(3, dtype=dtype), constant
    )

    check_tensor_array(
        identity(3, dtype=dtype, constant=constant),
        np.identity(3, dtype=dtype),
        constant,
    )

    check_tensor_array(
        ones((4, 5, 6), dtype=dtype, constant=constant),
        np.ones((4, 5, 6), dtype=dtype),
        constant,
    )

    check_tensor_array(
        ones_like(x, dtype=dtype, constant=constant),
        np.ones_like(x, dtype=dtype),
        constant,
    )

    check_tensor_array(
        ones_like(Tensor(x), dtype=dtype, constant=constant),
        np.ones_like(x, dtype=dtype),
        constant,
    )

    check_tensor_array(
        zeros((4, 5, 6), dtype=dtype, constant=constant),
        np.zeros((4, 5, 6), dtype=dtype),
        constant,
    )

    check_tensor_array(
        zeros_like(x, dtype=dtype, constant=constant),
        np.zeros_like(x, dtype=dtype),
        constant,
    )

    check_tensor_array(
        zeros_like(Tensor(x), dtype=dtype, constant=constant),
        np.zeros_like(x, dtype=dtype),
        constant,
    )

    check_tensor_array(
        full((4, 5, 6), 5.0, dtype=dtype, constant=constant),
        np.full((4, 5, 6), 5.0, dtype=dtype),
        constant,
    )

    check_tensor_array(
        full_like(x, 5.0, dtype=dtype, constant=constant),
        np.full_like(x, 5.0, dtype=dtype),
        constant,
    )

    check_tensor_array(
        full_like(Tensor(x), 5.0, dtype=dtype, constant=constant),
        np.full_like(x, 5.0, dtype=dtype),
        constant,
    )

    check_tensor_array(
        arange(3, 7, dtype=dtype, constant=constant),
        np.arange(3, 7, dtype=dtype),
        constant,
    )

    check_tensor_array(
        linspace(3, 7, dtype=dtype, constant=constant),
        np.linspace(3, 7, dtype=dtype),
        constant,
    )

    check_tensor_array(
        logspace(3, 7, dtype=dtype, constant=constant),
        np.logspace(3, 7, dtype=dtype),
        constant,
    )

    check_tensor_array(
        geomspace(3, 7, dtype=dtype, constant=constant),
        np.geomspace(3, 7, dtype=dtype),
        constant,
    )


@given(t=tensors(dtype=hnp.floating_dtypes(), include_grad=st.booleans()), in_graph=st.booleans())
def test_astensor_returns_tensor_reference_consistently(t: Tensor, in_graph: bool):
    if in_graph:
        t = +t
    assert astensor(t) is t
    assert astensor(t).grad is t.grad
    assert astensor(t).creator is t.creator

    assert astensor(t, dtype=t.dtype) is t
    assert astensor(t, constant=t.constant) is t
    assert astensor(t, dtype=t.dtype, constant=t.constant) is t
