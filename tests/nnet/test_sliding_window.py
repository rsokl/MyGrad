import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from mygrad.nnet.layers.operations.utils import sliding_window_view

dtype_strat_numpy = st.sampled_from(
    (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
)


@pytest.mark.parametrize(
    "args",
    [
        dict(step=None),
        dict(step=st.integers(max_value=-1)),
        dict(window_shape=st.none() | st.just(1)),
        dict(
            window_shape=st.lists(st.just(1), min_size=3)
        ),  # more window dims than arr
        dict(
            window_shape=(
                st.just((-1, 1))
                | st.just((-1, 1))
                | st.tuples(st.floats(), st.floats())
            )
        ),
        dict(
            window_shape=st.lists(st.integers(5, 7), min_size=1, max_size=2).filter(
                lambda x: 7 in x
            )
        ),  # window dim too large
        dict(dilation=st.sampled_from([-1, (1, 0), "aa", (1, 1, 1), 1.0])),
        dict(dilation=st.sampled_from([7, (1, 7), (10, 1)])),
    ],
)
@settings(deadline=None)
@given(data=st.data())
def test_input_validation(args: dict, data: st.DataObject):
    kwargs = dict(
        arr=np.arange(36).reshape(6, 6), window_shape=(1, 1), step=1, dilation=None
    )
    kwargs.update(
        (k, (data.draw(v, label=k)) if isinstance(v, st.SearchStrategy) else v)
        for k, v in args.items()
    )

    with pytest.raises((ValueError, TypeError)):
        sliding_window_view(**kwargs)


@settings(deadline=None)
@given(
    data=st.data(),
    x=hnp.arrays(
        dtype=dtype_strat_numpy,
        shape=hnp.array_shapes(max_dims=5, min_dims=1, max_side=20),
    ),
)
def test_sliding_window(data, x):
    """Test variations of window-shape, step, and dilation for sliding window
    view of N-dimensional array."""

    win_dim = data.draw(st.integers(1, x.ndim), label="win_dim")
    win_shape = data.draw(
        st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])), label="win_shape"
    )
    step = data.draw(
        st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])), label="step"
    )

    max_dilation = np.array(x.shape[-win_dim:]) // win_shape
    dilation = data.draw(
        st.one_of(
            st.none()
            | st.integers(1, min(max_dilation))
            | st.tuples(*(st.integers(1, s) for s in max_dilation))
        ),
        label="dilation",
    )
    y = sliding_window_view(x, window_shape=win_shape, step=step, dilation=dilation)

    if dilation is None:
        dilation = np.ones((len(win_shape),), dtype=int)

    if isinstance(dilation, int):
        dilation = np.full((len(win_shape),), fill_value=dilation, dtype=int)

    for ind in np.ndindex(*y.shape[:win_dim]):
        slices = tuple(
            slice(i * s, i * s + w * d, d)
            for i, w, s, d in zip(ind, win_shape, step, dilation)
        )
        assert_allclose(actual=y[tuple([*ind])], desired=x[(..., *slices)])


@given(dtype=dtype_strat_numpy)
def test_memory_details(dtype):
    """Ensure that:
    - function handles non C-contiguous layouts correctly
    - output is view of input
    - output is not writeable"""
    x = np.arange(20).reshape(2, 10).astype(dtype)
    x = np.asfortranarray(x)
    y = sliding_window_view(x, (5,), 5)
    soln = np.array(
        [
            [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]],
            [[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]],
        ]
    )
    assert not y.flags["WRITEABLE"]
    assert_allclose(actual=y, desired=soln)

    x = np.arange(20).reshape(2, 10)
    x = np.ascontiguousarray(x)
    y = sliding_window_view(x, (5,), 5)
    assert not y.flags["WRITEABLE"]
    assert np.shares_memory(x, y)
    assert_allclose(actual=y, desired=soln)
