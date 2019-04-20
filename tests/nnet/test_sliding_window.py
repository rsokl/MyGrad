from mygrad.nnet.layers.utils import sliding_window_view

import numpy as np
from numpy.testing import assert_allclose

import hypothesis.strategies as st
from hypothesis import given, settings
import hypothesis.extra.numpy as hnp


dtype_strat_numpy = st.sampled_from((np.int8, np.int16, np.int32, np.int64,
                                     np.float16, np.float32, np.float64))


@settings(deadline=None)
@given(data=st.data(),
       x=hnp.arrays(dtype=dtype_strat_numpy, shape=hnp.array_shapes(max_dims=5, min_dims=1, max_side=20)))
def test_sliding_window(data, x):
    """ Test variations of window-shape, step, and dilation for sliding window
        view of N-dimensional array."""

    win_dim = data.draw(st.integers(1, x.ndim), label="win_dim")
    win_shape = data.draw(st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])),
                          label="win_shape")
    step = data.draw(st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])),
                     label="step")

    max_dilation = np.array(x.shape[-win_dim:]) // win_shape
    dilation = data.draw(st.one_of(st.none(), st.tuples(*(st.integers(1, s) for s in max_dilation))),
                         label="dilation")
    y = sliding_window_view(x, window_shape=win_shape, step=step, dilation=dilation)

    if dilation is None:
        dilation = np.ones((len(win_shape),), dtype=int)

    for ind in np.ndindex(*y.shape[:win_dim]):
        slices = tuple(slice(i*s, i*s + w*d, d) for i,w,s,d in zip(ind, win_shape, step, dilation))
        assert_allclose(actual=y[tuple([*ind])], desired=x[(..., *slices)])


@given(dtype=dtype_strat_numpy)
def test_memory_details(dtype):
    """ Ensure that:
          - function handles non C-contiguous layouts correctly
          - output is view of input
          - output is not writeable"""
    x = np.arange(20).reshape(2, 10).astype(dtype)
    x = np.asfortranarray(x)
    y = sliding_window_view(x, (5,), 5)
    soln = np.array([[[ 0,  1,  2,  3,  4],
                      [10, 11, 12, 13, 14]],
                     [[ 5,  6,  7,  8,  9],
                      [15, 16, 17, 18, 19]]])
    assert not y.flags["WRITEABLE"]
    assert_allclose(actual=y, desired=soln)

    x = np.arange(20).reshape(2, 10)
    x = np.ascontiguousarray(x)
    y = sliding_window_view(x, (5,), 5)
    assert not y.flags["WRITEABLE"]
    assert np.shares_memory(x, y)
    assert_allclose(actual=y, desired=soln)

