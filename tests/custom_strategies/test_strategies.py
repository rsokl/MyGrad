from numbers import Real
from typing import List, Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note, settings
from numpy.testing import assert_array_equal

from mygrad import Tensor
from tests.custom_strategies import (
    _factors,
    adv_integer_index,
    arbitrary_indices,
    basic_indices,
    choices,
    integer_index,
    slice_index,
    tensors,
    valid_axes,
    valid_shapes,
)


@given(seq=st.lists(elements=st.integers()), replace=st.booleans(), data=st.data())
def test_choices(seq: List[int], replace: bool, data: st.SearchStrategy):
    """ Ensures that the `choices` strategy:
        - draws from the provided sequence
        - respects input parameters"""
    upper = len(seq) + 10 if replace and seq else len(seq)
    size = data.draw(st.integers(0, upper), label="size")
    chosen = data.draw(choices(seq, size=size, replace=replace), label="choices")
    assert set(chosen) <= set(seq), (
        "choices contains elements that do not " "belong to `seq`"
    )
    assert len(chosen) == size, "the number of choices does not match `size`"

    if not replace and len(set(seq)) == len(seq):
        unique_choices = sorted(set(chosen))
        assert unique_choices == sorted(chosen), (
            "`choices` with `replace=False` draws " "elements with replacement"
        )


@given(size=st.integers(1, 10), data=st.data())
def test_integer_index(size: int, data: st.SearchStrategy):
    index = data.draw(integer_index(size), label="index")
    x = np.empty((size,))
    o = x[index]  # raises if invalid index
    assert isinstance(
        o, Real
    ), "An integer index should produce a number from a 1D array"


@given(size=st.integers(1, 10), data=st.data())
def test_slice_index(size: int, data: st.SearchStrategy):
    index = data.draw(slice_index(size), label="index")
    x = np.empty((size,))
    o = x[index]  # raises if invalid index
    assert isinstance(o, np.ndarray) and o.ndim == 1, (
        "A slice index should produce " "a 1D array from a 1D array"
    )
    if o.size:
        assert np.shares_memory(o, x), "A slice should produce a view of `x`"

    if index.start is not None:
        assert -size <= index.start <= size

    if index.stop is not None:
        assert -size <= index.stop <= size


@given(shape=hnp.array_shapes(min_dims=3), data=st.data())
def test_basic_index(shape: Tuple[int, ...], data: st.SearchStrategy):
    min_dim = data.draw(st.integers(0, len(shape) + 2), label="min_dim")
    max_dim = data.draw(st.integers(min_dim, min_dim + len(shape)), label="max_dim")
    index = data.draw(
        basic_indices(shape=shape, min_dims=min_dim, max_dims=max_dim), label="index"
    )
    x = np.zeros(shape, dtype=int)
    o = x[index]  # raises if invalid index

    note(f"`x[index]`: {o}")
    if o.size and o.ndim > 0:
        assert np.shares_memory(x, o), (
            "The basic index should produce a " "view of the original array."
        )
    assert min_dim <= o.ndim <= max_dim, (
        "The dimensionality input constraints " "were not obeyed"
    )


@given(shape=hnp.array_shapes(min_dims=1), min_dims=st.integers(1, 3), data=st.data())
def test_advanced_integer_index(
    shape: Tuple[int, ...], min_dims: int, data: st.SearchStrategy
):
    max_dims = data.draw(st.integers(min_dims, min_dims + 3), label="max_dims")
    index = data.draw(adv_integer_index(shape, min_dims=min_dims, max_dims=max_dims))
    x = np.zeros(shape)
    out = x[index]  # raises if the index is invalid
    note(f"x[index]: {out}")
    assert min_dims <= out.ndim <= max_dims, "The input parameters were not respected"
    assert not np.shares_memory(
        x, out
    ), "An advanced index should create a copy upon indexing"


@given(
    shape=hnp.array_shapes(min_dims=1, max_dims=5),
    data=st.data(),
    permit_none=st.booleans(),
)
def test_valid_single_axis(shape, data, permit_none):
    axis = data.draw(
        valid_axes(ndim=len(shape), single_axis_only=True, permit_none=permit_none),
        label="axis",
    )
    x = np.empty(shape)
    np.argmin(x, axis=axis)  # raises if `axis` is invalid

    if not permit_none:
        assert axis is not None


@given(
    shape=hnp.array_shapes(min_dims=1, max_dims=5),
    data=st.data(),
    permit_none=st.booleans(),
    pos_only=st.booleans(),
)
def test_valid_axes(shape, data, permit_none, pos_only):
    min_dim = data.draw(st.integers(0, len(shape)), label="min_dim")
    max_dim = data.draw(
        st.one_of(st.none(), st.integers(min_dim, len(shape))), label="max_dim"
    )
    axis = data.draw(
        valid_axes(
            ndim=len(shape),
            permit_none=permit_none,
            pos_only=pos_only,
            min_dim=min_dim,
            max_dim=max_dim,
        ),
        label="axis",
    )
    x = np.zeros(shape)
    np.sum(x, axis=axis)
    if not permit_none:
        assert axis is not None

    if pos_only and axis is not None:
        if isinstance(axis, tuple):
            assert all(i >= 0 for i in axis)
        else:
            assert axis >= 0

    if axis is not None:
        if isinstance(axis, tuple):
            assert min_dim <= len(axis)

            if max_dim is not None:
                assert len(axis) <= max_dim
        else:
            assert min_dim <= 1


@given(st.integers(min_value=0, max_value=1000))
def test_factors(size: int):
    factors = _factors(size)
    a_factors = np.array(factors)
    assert len(set(a_factors)) == len(a_factors)
    assert_array_equal(a_factors * a_factors[::-1], size)


@given(
    arr=hnp.arrays(
        dtype=bool, fill=st.just(False), shape=hnp.array_shapes(min_dims=0, max_dims=10)
    ),
    data=st.data(),
)
def test_valid_shapes(arr: np.ndarray, data: st.DataObject):
    newshape = data.draw(valid_shapes(arr.size, min_len=0), label="newshape")
    arr.reshape(newshape)


@settings(deadline=None)
@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(min_side=0, max_side=4, min_dims=0, max_dims=5),
        dtype=float,
    ),
    data=st.data(),
)
def test_arbitrary_indices_strategy(a, data):
    shape = a.shape
    index = data.draw(arbitrary_indices(shape))

    # if index does not comply with numpy indexing
    # rules, numpy will raise an error
    a[index]


@pytest.mark.parametrize("constant", [st.booleans(), None])
def test_tensors_handles_constant_strat(constant):
    constants = []
    kwargs = dict(dtype=np.int8, shape=(2, 3))
    if constant is not None:
        kwargs["constant"] = constant

    @given(x=tensors(**kwargs))
    def f(x):
        constants.append(x.constant)

    f()

    assert len(set(constants)) > 1


@pytest.mark.parametrize("constant", [True, False])
@given(data=st.data())
def test_tensors_static_constant(constant: bool, data: st.DataObject):
    tensor = data.draw(tensors(np.int8, (2, 3), constant=constant), label="tensor")
    assert isinstance(tensor, Tensor)
    assert tensor.constant is constant
    assert tensor.grad is None


@given(data=st.data(), shape=hnp.array_shapes())
def test_tensors_shape(shape, data: st.DataObject):
    tensor = data.draw(tensors(np.int8, shape=shape), label="tensor")
    assert isinstance(tensor, Tensor)
    assert tensor.shape == shape
    assert tensor.grad is None


@given(data=st.data(), dtype=hnp.floating_dtypes() | hnp.integer_dtypes())
def test_tensors_dtype(dtype, data: st.DataObject):
    tensor = data.draw(tensors(dtype=dtype, shape=(2, 3)), label="tensor")
    assert isinstance(tensor, Tensor)
    assert tensor.dtype == dtype
    assert tensor.grad is None


@given(
    data=st.data(),
    dtype=hnp.floating_dtypes(),
    shape=hnp.array_shapes(min_dims=0, min_side=0),
    grad_dtype=hnp.floating_dtypes() | st.none(),
    grad_elements_bounds=st.just((100, 200)) | st.none(),
)
def test_tensors_with_grad(
    dtype, data: st.DataObject, shape, grad_dtype, grad_elements_bounds
):
    tensor = data.draw(
        tensors(
            dtype=dtype,
            shape=shape,
            include_grad=True,
            grad_dtype=grad_dtype,
            grad_elements_bounds=grad_elements_bounds,
        ),
        label="tensor",
    )
    assert isinstance(tensor, Tensor)
    assert tensor.dtype == dtype
    assert isinstance(tensor.grad, np.ndarray)
    assert tensor.grad.shape == tensor.shape
    assert tensor.grad.dtype == (grad_dtype if grad_dtype is not None else tensor.dtype)
    if grad_elements_bounds is not None:
        assert np.all((100 <= tensor.grad) & (tensor.grad <= 200))
    else:
        assert np.all((-10 <= tensor.grad) & (tensor.grad <= 10))
