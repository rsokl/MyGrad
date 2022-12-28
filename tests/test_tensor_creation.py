from copy import deepcopy
from functools import partial
from typing import Callable, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, infer, settings, note
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import Tensor, astensor, mem_guard_off
from mygrad.operation_base import _NoValue
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
from mygrad.typing import ArrayLike, DTypeLikeReals, Shape
from tests.custom_strategies import real_dtypes, tensors, valid_constant_arg
from tests.utils.checkers import expected_constant
from tests.utils.functools import SmartSignature


def check_tensor_array(tensor, array, data_compare=True):
    assert isinstance(tensor, Tensor)
    if data_compare:
        assert_array_equal(tensor.data, array)
    assert tensor.dtype is array.dtype


def clamp(val, min_=0.1, max_=None):
    if val is not _NoValue:
        val = max(val, min_)
        if max_ is not None:
            val = min(val, max_)
    return val


@pytest.mark.parametrize("as_kwargs", [True, False])
@pytest.mark.parametrize(
    "mygrad_func, numpy_func",
    [
        (arange, np.arange),
        (linspace, np.linspace),
        (geomspace, np.geomspace),
        (logspace, np.logspace),
    ],
)
@settings(max_examples=200)
@given(
    start=st.just(_NoValue) | st.integers(min_value=-10, max_value=10),
    stop=st.integers(min_value=-10, max_value=10),
    step=st.just(_NoValue) | st.integers(0, 5),
    dtype=infer,
    data=st.data(),
)
def test_arange_like_against_numpy_equivalent(
    start,
    stop,
    step,
    dtype: DTypeLikeReals,
    mygrad_func: Callable,
    numpy_func: Callable,
    data: st.DataObject,
    as_kwargs: bool,
):
    if numpy_func is not np.arange:
        axis = data.draw(st.sampled_from([_NoValue, -1]), label="axis")
    else:
        axis = _NoValue

    if numpy_func is np.geomspace:
        start = clamp(start)
        stop = clamp(stop)
        step = clamp(step)

    if numpy_func is np.logspace:
        start = clamp(start,max_=3)
        stop = clamp(stop,max_=3)
        step = clamp(step)

    if as_kwargs:
        inputs = SmartSignature(
            start=start, stop=stop, step=step, dtype=dtype, axis=axis
        )
    else:
        inputs = SmartSignature(start, stop, step, dtype=dtype, axis=axis)
    try:
        array = numpy_func(*inputs, **inputs)
    except (ZeroDivisionError, TypeError) as e:
        with pytest.raises(type(e)):
            mygrad_func(*inputs, **inputs)
        return

    constant = data.draw(valid_constant_arg(array.dtype), label="constant")
    tensor = mygrad_func(*inputs, **inputs, constant=constant)

    assert_array_equal(tensor, array)
    assert tensor.dtype == array.dtype
    assert tensor.constant is expected_constant(
        dest_dtype=tensor.dtype, constant=constant
    )


@pytest.mark.parametrize(
    "mygrad_func, numpy_func",
    [
        (empty, np.empty),
        (ones, np.ones),
        (zeros, np.zeros),
        (partial(full, fill_value=2), partial(np.full, fill_value=2)),
    ],
)
@settings(max_examples=20)
@given(shape=infer, dtype=infer, data=st.data())
def test_tensor_creation_from_shape_against_numpy_equivalent(
    shape: Shape,
    dtype: DTypeLikeReals,
    data: st.DataObject,
    mygrad_func: Callable,
    numpy_func: Callable,
):
    array = numpy_func(shape, dtype=dtype)

    constant = data.draw(valid_constant_arg(array.dtype), label="constant")
    tensor = mygrad_func(shape, dtype=dtype, constant=constant)
    if numpy_func is not np.empty:
        assert_array_equal(tensor, array)
    assert tensor.dtype == array.dtype
    assert tensor.constant is expected_constant(
        dest_dtype=tensor.dtype, constant=constant
    )


@pytest.mark.parametrize(
    "mygrad_func, numpy_func",
    [
        (partial(full_like, fill_value=-2), partial(np.full_like, fill_value=-2)),
        (zeros_like, np.zeros_like),
        (ones_like, np.ones_like),
        (empty_like, np.empty_like),
    ],
)
@given(arr_like=infer, dtype=infer, data=st.data())
def test_tensor_create_like_against_numpy_equivalent(
    arr_like: ArrayLike,
    dtype: DTypeLikeReals,
    data: st.DataObject,
    mygrad_func: Callable,
    numpy_func: Callable,
):
    _flat_shape = (np.asarray(arr_like).size,)
    shape = data.draw(st.sampled_from([_NoValue, _flat_shape]), label="shape")
    inputs = SmartSignature(arr_like, dtype=dtype, shape=shape)

    array = numpy_func(*inputs, **inputs)

    constant = data.draw(valid_constant_arg(array.dtype), label="constant")
    tensor = mygrad_func(*inputs, **inputs, constant=constant)

    if numpy_func is not np.empty_like:
        assert_array_equal(tensor, array)
    assert tensor.dtype == array.dtype
    assert tensor.constant is expected_constant(
        arr_like, dest_dtype=tensor.dtype, constant=constant
    )


@given(dtype=st.just(_NoValue) | real_dtypes, data=st.data(), n=st.integers(0, 4))
def test_identity(dtype: DTypeLikeReals, data: st.DataObject, n: int):
    inputs = SmartSignature(n, dtype=dtype)
    arr = np.identity(*inputs, **inputs)

    constant = data.draw(valid_constant_arg(arr.dtype), label="constant")
    tensor = identity(*inputs, **inputs, constant=constant)

    assert_array_equal(tensor, arr)
    assert tensor.dtype == arr.dtype
    assert tensor.constant is expected_constant(
        dest_dtype=tensor.dtype, constant=constant
    )


@given(
    dtype=st.just(_NoValue) | real_dtypes,
    data=st.data(),
    n=st.integers(0, 4),
    m=st.integers(0, 4),
)
def test_eye(dtype: DTypeLikeReals, data: st.DataObject, n: int, m: int):
    k = data.draw((st.just(_NoValue) | st.integers(0, min(n, m))), label="constant")
    inputs = SmartSignature(n, m, k, dtype=dtype)
    arr = np.eye(*inputs, **inputs)

    constant = data.draw(valid_constant_arg(arr.dtype), label="constant")
    tensor = eye(*inputs, **inputs, constant=constant)

    assert_array_equal(tensor, arr)
    assert tensor.dtype == arr.dtype
    assert tensor.constant is expected_constant(
        dest_dtype=tensor.dtype, constant=constant
    )


def test_simple_as_tensor():
    arr = np.array(1, dtype=np.int8)
    x = astensor(arr)
    assert x.constant is True
    assert x.data.dtype == np.int8
    assert x.data.item() == 1
    assert x.data is arr

    arr = np.array(1.0, dtype=np.float32)
    x = astensor(arr)
    assert x.constant is False
    assert x.data.dtype == np.float32
    assert x.data.item() == 1.0
    assert x.data is arr


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


general_arrays = hnp.arrays(
    shape=hnp.array_shapes(min_side=0, min_dims=0),
    dtype=hnp.floating_dtypes() | hnp.integer_dtypes(),
    elements=st.integers(-2, 2),
)


class NotSet:
    pass


not_set = st.just(NotSet)


@settings(max_examples=1000)
@given(
    arr_like=general_arrays.map(lambda x: x.tolist()) | general_arrays,
    dtype=not_set | st.none() | hnp.floating_dtypes() | hnp.integer_dtypes(),
    copy=not_set | st.booleans(),
    # can't generically test with `constant=False` because of int-dtypes
    constant=st.none() | st.just(True),
    ndmin=not_set | st.integers(-6, 6),
)
def test_tensor_mirrors_array(arr_like, dtype, copy, constant, ndmin):
    if isinstance(arr_like, np.ndarray):
        note(f"arr_like.dtype: {arr_like.dtype}")

    kwargs = {}
    for name, var_ in [("dtype", dtype), ("copy", copy), ("ndmin", ndmin)]:
        if var_ is not NotSet:
            kwargs[name] = var_

    try:
        arr = np.array(arr_like, **kwargs)
    except (ValueError, OverflowError):
        assume(False)
        return

    tensor_like = (
        Tensor(arr_like.copy(), constant=constant)
        if isinstance(arr_like, np.ndarray)
        else arr_like
    )

    tens = mg.tensor(tensor_like, constant=constant, **kwargs)

    assert tens.dtype == arr.dtype
    assert tens.shape == arr.shape
    assert np.shares_memory(tens, tensor_like) is np.shares_memory(arr, arr_like)

    if arr.dtype.byteorder != ">":
        # condition due to https://github.com/numpy/numpy/issues/22897

        assert (tens is tensor_like) is (arr is arr_like)
        assert (tens.base is tensor_like) is (arr.base is arr_like)
        if tens.base is None:
            # sometimes numpy makes internal views; mygrad should never do this
            assert arr.base is not arr_like


@given(
    t=tensors(dtype=hnp.floating_dtypes() | hnp.integer_dtypes()),
    dtype=st.none() | hnp.floating_dtypes() | hnp.integer_dtypes(),
    copy=st.booleans(),
    # can't generically test with `constant=False` because of int-dtypes
    constant=st.none() | st.just(True),
    ndmin=st.none() | st.floats(),
)
def test_bad_ndmin_raises(t, dtype, copy, constant, ndmin):
    with pytest.raises(TypeError):
        mg.tensor(t, dtype=dtype, copy=copy, ndmin=ndmin, constant=constant)


@given(
    shapes=hnp.mutually_broadcastable_shapes(
        num_shapes=1, base_shape=(4, 3, 2), max_dims=3, max_side=4
    ),
    fill_is_tensor=st.booleans(),
    data=st.data(),
)
def test_full_can_broadcast_fill_value(
    shapes: hnp.BroadcastableShapes, fill_is_tensor: bool, data: st.DataObject
):
    fill_value = data.draw(hnp.arrays(shape=shapes.input_shapes[0], dtype=float))
    expected = np.full(shape=shapes.result_shape, fill_value=fill_value)

    if fill_is_tensor:
        fill_value = Tensor(fill_value)

    actual = full(shape=shapes.result_shape, fill_value=fill_value)

    assert_array_equal(actual, expected)
