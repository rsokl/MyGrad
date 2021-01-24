from copy import deepcopy
from typing import Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from numpy.testing import assert_array_equal

from mygrad import Tensor, astensor, mem_guard_off
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
from tests.custom_strategies import tensors, valid_constant_arg


def check_tensor_array(tensor, array, constant, data_compare=True):
    if constant is None:
        constant = not np.issubdtype(tensor.dtype, np.floating)
    assert isinstance(tensor, Tensor)
    if data_compare:
        assert_array_equal(tensor.data, array)
    assert tensor.dtype is array.dtype
    assert tensor.constant is constant


ref_arr = np.arange(3)


@pytest.mark.parametrize(
    "mygrad_func, numpy_func, args",
    [
        (empty, np.empty, [(3, 2)]),
        (empty_like, np.empty_like, [ref_arr]),
        (eye, np.eye, [3]),
        (identity, np.identity, [5]),
        (ones, np.ones, [(4, 3, 2)]),
        (ones_like, np.ones_like, [ref_arr]),
        (zeros, np.zeros, [(4, 3, 2)]),
        (zeros_like, np.zeros_like, [ref_arr]),
        (full, np.full, [(4, 3, 2), 5.0]),
        (full_like, np.full_like, [ref_arr, 5.0]),
        (arange, np.arange, [3, 7]),
        (linspace, np.linspace, [3, 7]),
        (geomspace, np.geomspace, [3, 7]),
        (logspace, np.logspace, [3, 7]),
    ],
)
@given(data=st.data(), dtype=hnp.floating_dtypes() | hnp.integer_dtypes())
def test_tensor_creation_matches_array_creation(
    mygrad_func, numpy_func, args, data: st.DataObject, dtype: np.dtype
):
    constant = data.draw(valid_constant_arg(dtype), label="constant")

    check_data = mygrad_func not in {empty, empty_like}

    kwargs = {} if dtype is None else dict(dtype=dtype)

    check_tensor_array(
        mygrad_func(*args, constant=constant, **kwargs),
        numpy_func(*args, **kwargs),
        constant,
        data_compare=check_data,
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
    assert t2.data is t.data
    assert t2.creator is None

    t3 = astensor(t, dtype=t.dtype, constant=not t.constant)
    assert t3 is not t
    assert t3.data is t.data
    assert t3.creator is None


@given(
    t=tensors(dtype=hnp.floating_dtypes(), include_grad=st.booleans()),
    in_graph=st.booleans(),
    dtype=st.none() | hnp.floating_dtypes(),
    constant=st.none() | st.booleans(),
)
@mem_guard_off
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
        elements=dict(min_value=0, max_value=0),
    )
    | tensors(
        dtype=hnp.floating_dtypes(),
        include_grad=st.booleans(),
        elements=dict(min_value=0, max_value=0),
    ),
    as_list=st.booleans(),
    dtype=st.none() | hnp.integer_dtypes() | hnp.floating_dtypes(),
    data=st.data(),
)
def test_as_tensor(
    a: Union[np.ndarray, Tensor], as_list: bool, dtype, data: st.DataObject
):
    """Ensures `astensor` produces a tensor with the expected data, dtype, and constant,
    and that it doesn't mutate the input."""
    assume(~np.any(np.isnan(a)))

    constant = data.draw(
        valid_constant_arg(np.array(a, dtype=dtype).dtype), label="constant"
    )

    # make copies to check mutations
    if as_list:
        a = np.asarray(a).tolist()
        original = deepcopy(a)
    else:
        original = a.copy()

    t = astensor(a, dtype=dtype, constant=constant)

    ref_tensor = a if t is a else Tensor(a, dtype=dtype, constant=constant)

    assert isinstance(t, Tensor)
    assert t.dtype == ref_tensor.dtype
    assert t.constant is ref_tensor.constant
    assert_array_equal(ref_tensor.data, t.data)

    if as_list:
        assert a == original, "the original array was mutated"
    else:
        assert_array_equal(original, a, err_msg="the original array was mutated")
