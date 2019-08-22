from numbers import Real
from typing import List, Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, note

from tests.custom_strategies import (
    adv_integer_index,
    basic_index,
    choices,
    integer_index,
    slice_index,
    valid_axes,
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
        basic_index(shape=shape, min_dim=min_dim, max_dim=max_dim), label="index"
    )
    x = np.zeros(shape, dtype=int)
    o = x[index]  # raises if invalid index

    note("`x[index]`: {}".format(o))
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
    note("x[index]: {}".format(out))
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
    x = np.empty(shape)
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
