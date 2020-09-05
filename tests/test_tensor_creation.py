from copy import deepcopy
from typing import Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
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
from tests.custom_strategies import tensors


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


@given(
    t=tensors(dtype=hnp.floating_dtypes(), include_grad=st.booleans()),
    in_graph=st.booleans(),
)
def test_astensor_returns_tensor_reference_consistently(t: Tensor, in_graph: bool):
    if in_graph:
        t = +t
    assert astensor(t) is t
    assert astensor(t, dtype=t.dtype) is t
    assert astensor(t, constant=t.constant) is t
    assert astensor(t, dtype=t.dtype, constant=t.constant) is t


@given(
    t=tensors(dtype=hnp.floating_dtypes(), include_grad=st.booleans()),
    in_graph=st.booleans(),
)
def test_astensor_with_incompat_constant_still_passes_array_ref(
    t: Tensor, in_graph: bool
):
    if in_graph:
        t = +t

    t2 = astensor(t, constant=not t.constant)
    assert t2 is not t
    assert (t2.data is t.data) is t2.constant  # data copied if constant=False
    assert t2.creator is None

    t3 = astensor(t, dtype=t.dtype, constant=not t.constant)
    assert t3 is not t
    assert (t3.data is t.data) is t3.constant  # data copied if constant=False
    assert t3.creator is None


@given(
    t=tensors(dtype=hnp.floating_dtypes(), include_grad=st.booleans()),
    in_graph=st.booleans(),
    dtype=st.none() | hnp.floating_dtypes(),
    constant=st.none() | st.booleans(),
)
def test_astensor_doesnt_mutate_input_tensor(
    t: Tensor, in_graph: bool, dtype, constant: bool
):
    if in_graph:
        t = +t
    o_constant = t.constant
    o_creator = t.creator
    o_data = t.data.copy()
    o_grad = None if t.grad is None else t.grad.copy()

    astensor(t, dtype=dtype, constant=constant)
    assert t.constant is o_constant
    assert t.creator is o_creator
    assert_array_equal(t, o_data)
    if o_grad is not None:
        assert_array_equal(t.grad, o_grad)
    else:
        assert t.grad is None


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        dtype=hnp.integer_dtypes() | hnp.floating_dtypes(),
    )
    | tensors(dtype=hnp.floating_dtypes(), include_grad=st.booleans()),
    as_list=st.booleans(),
    dtype=st.none() | hnp.floating_dtypes(),
    constant=st.none() | st.booleans(),
)
def test_as_tensor(a: Union[np.ndarray, Tensor], as_list: bool, dtype, constant: bool):
    """Ensures `astensor` produces a tensor with the expected data, dtype, and constant,
    and that it doesn't mutate the input."""
    assume(~np.any(np.isnan(a)))
    # make copies to check mutations
    if as_list:
        a = np.asarray(a).tolist()
        original = deepcopy(a)
    else:
        original = a.copy()

    expected_dtype = dtype if dtype is not None else np.asarray(a).dtype

    if constant is not None:
        expected_constant = constant
    elif isinstance(a, Tensor):
        expected_constant = a.constant
    else:
        expected_constant = False

    t = astensor(a, dtype=dtype, constant=constant)

    assert isinstance(t, Tensor)
    assert t.dtype == expected_dtype
    assert t.constant is expected_constant
    assert_array_equal(np.asarray(a, dtype=dtype), t.data)

    if as_list:
        assert a == original, "the original array was mutated"
    else:
        assert_array_equal(original, a, err_msg="the original array was mutated")
