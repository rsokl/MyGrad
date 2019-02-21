from tests.custom_strategies import broadcastable_shape, choices, integer_index
from tests.custom_strategies import slice_index, basic_index, adv_integer_index

from hypothesis import given, note
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from numbers import Real

import numpy as np
from typing import Tuple, List


@given(seq=st.lists(elements=st.integers()),
       replace=st.booleans(),
       data=st.data())
def test_choices(seq: List[int], replace: bool, data: st.SearchStrategy):
    """ Ensures that the `choices` strategy:
        - draws from the provided sequence
        - respects input parameters"""
    upper = len(seq) + 10 if replace and seq else len(seq)
    size = data.draw(st.integers(0, upper), label="size")
    chosen = data.draw(choices(seq, size=size, replace=replace), label="choices")
    assert set(chosen) <= set(seq), "choices contains elements that do not " \
                                    "belong to `seq`"
    assert len(chosen) == size, "the number of choices does not match `size`"

    if not replace and len((set(seq))) == len(seq):
        unique_choices = sorted(set(chosen))
        assert unique_choices == sorted(chosen) , "`choices` with `replace=False` draws " \
                                                  "elements with replacement"


@given(size=st.integers(1, 10),
       data=st.data())
def test_integer_index(size: int, data: st.SearchStrategy):
    index = data.draw(integer_index(size), label="index")
    x = np.empty((size,))
    o = x[index]  # raises if invalid index
    assert isinstance(o, Real), "An integer index should produce a number from a 1D array"


@given(size=st.integers(1, 10),
       data=st.data())
def test_slice_index(size: int, data: st.SearchStrategy):
    index = data.draw(slice_index(size), label="index")
    x = np.empty((size,))
    o = x[index]  # raises if invalid index
    assert isinstance(o, np.ndarray) and o.ndim == 1, "A slice index should produce " \
                                                      "a 1D array from a 1D array"
    if o.size:
        assert np.shares_memory(o, x), "A slice should produce a view of `x`"

    if index.start is not None:
        assert -size <= index.start <= size

    if index.stop is not None:
        assert -size <= index.stop <= size


@given(shape=hnp.array_shapes(), allow_singleton=st.booleans(),
       min_dim=st.integers(0, 6),
       min_side=st.integers(1, 6),
       data=st.data())
def test_broadcast_compat_shape(shape: Tuple[int, ...],
                                allow_singleton: bool,
                                min_dim: int,
                                min_side: int,
                                data: st.SearchStrategy):
    """ Ensures that the `broadcastable_shape` strategy:
        - produces broadcastable shapes
        - respects input parameters"""
    max_side = data.draw(st.integers(min_side, min_side + 5), label="max side")
    max_dim = data.draw(st.integers(min_dim, max(min_dim, len(shape)+3)), label="max dim")
    compat_shape = data.draw(broadcastable_shape(shape=shape, allow_singleton=allow_singleton,
                                                 min_dim=min_dim, max_dim=max_dim,
                                                 min_side=min_side, max_side=max_side),
                             label="broadcastable_shape")
    assert min_dim <= len(compat_shape) <= max_dim, \
        "a shape of inappropriate dimensionality was generated by the strategy"

    a = np.empty(shape)
    b = np.empty(compat_shape)
    np.broadcast(a, b)  # error if drawn shape for b is not broadcast-compatible

    if not allow_singleton:
        small_dim = min(a.ndim, b.ndim)
        if small_dim:
            assert shape[-small_dim:] == compat_shape[-small_dim:], \
                "singleton dimensions were included by the strategy"

    if len(compat_shape) > len(shape):
        n = len(compat_shape) - len(shape)
        for side in compat_shape[:n]:
            assert min_side <= side <= max_side, \
                "out-of-bound sides were generated by the strategy"


@given(shape=hnp.array_shapes(min_dims=3), data=st.data())
def test_basic_index(shape: Tuple[int, ...], data: st.SearchStrategy):
    min_dim = data.draw(st.integers(0, len(shape) + 2), label="min_dim")
    max_dim = data.draw(st.integers(min_dim, min_dim + len(shape)), label="max_dim")
    index = data.draw(basic_index(shape=shape, min_dim=min_dim, max_dim=max_dim), label="index")
    x = np.zeros(shape, dtype=int)
    o = x[index]  # raises if invalid index

    note("`x[index]`: {}".format(o))
    if o.size and o.ndim > 0:
        assert np.shares_memory(x, o), "The basic index should produce a " \
                                       "view of the original array."
    assert min_dim <= o.ndim <= max_dim, "The dimensionality input constraints " \
                                         "were not obeyed"


@given(shape=hnp.array_shapes(min_dims=1), min_dims=st.integers(1, 3), data=st.data())
def test_advanced_integer_index(shape: Tuple[int, ...], min_dims: int, data: st.SearchStrategy):
    max_dims = data.draw(st.integers(min_dims, min_dims+3), label="max_dims")
    index = data.draw(adv_integer_index(shape, min_dims=min_dims, max_dims=max_dims))
    x = np.zeros(shape)
    out = x[index]  # raises if the index is invalid
    note(f"x[index]: {out}")
    assert min_dims <= out.ndim <= max_dims, "The input parameters were not respected"
    assert not np.shares_memory(x, out), "An advanced index should create a copy upon indexing"
