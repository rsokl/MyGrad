import inspect

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note
from numpy.testing import assert_array_equal

import mygrad._graph_tracking as _tracking
from mygrad import Tensor, amax, no_autodiff
from tests.custom_strategies import tensors


def test_graph_tracking_is_on_by_default():
    assert _tracking.TRACK_GRAPH is True


@given(x=tensors(shape=(1,), elements=st.floats(-100, 100)))
def test_no_autodiff_context_manager(x: Tensor):

    with no_autodiff:
        y = +x

        assert y.creator is None
        assert y.data.flags.writeable

        assert not x._ops
        assert x.data.flags.writeable

        y.backward()  # should be no-op

    assert y.constant
    assert y.grad is None
    assert x.grad is None

    # check that standard behavior is restored
    y = 2 * x

    assert y.constant is x.constant
    assert y.creator is not None
    assert not y.data.flags.writeable

    assert x._ops
    assert not x.data.flags.writeable

    if not y.constant:
        y.backward()
        assert_array_equal(y.grad, np.full_like(y, 1.0))
        assert_array_equal(x.grad, np.full_like(x, 2.0))


@given(old_x=tensors(shape=(2,), constant=False, elements=st.floats(-100, 100)))
def test_no_tracking_for_inplace_op(old_x: Tensor):
    expected = old_x.copy()
    expected[0] = -1

    x = +old_x
    y = 2 * x

    with no_autodiff:
        x[0] = -1

    assert_array_equal(x, expected)

    y.backward()
    assert_array_equal(old_x.grad, np.full_like(old_x, 2.0))


def test_no_autodiff_context_manager_restores_state():

    with pytest.raises(ValueError):
        with no_autodiff:
            assert not _tracking.TRACK_GRAPH
            raise ValueError()

    assert _tracking.TRACK_GRAPH


@given(x=tensors(shape=(1,)))
def test_no_autodiff_decorator(x: Tensor):
    @no_autodiff
    def func(x):
        return +x

    y = func(x)
    assert y.constant
    assert y.creator is None
    assert not y._ops
    assert y.data.flags.writeable

    assert not x._ops
    assert x.data.flags.writeable


def test_no_autodiff_decorator_restores_state():
    @no_autodiff
    def func():
        assert not _tracking.TRACK_GRAPH
        raise ValueError()

    with pytest.raises(ValueError):
        func()

    assert _tracking.TRACK_GRAPH


def test_decorator_is_transparent_to_function_information():
    def undecorated(x, y, z=None, *, kwarg, **kwargs):
        """a very special docstring"""
        pass

    decorated = no_autodiff(undecorated)

    assert inspect.signature(undecorated) == inspect.signature(decorated)
    assert undecorated.__doc__ == decorated.__doc__
    assert undecorated.__name__ == decorated.__name__


@given(tensors(shape=(3, 2, 4), elements=st.floats(-100, 100)).map(np.asarray))
def test_to_numpy_returns_numpy_array(x: np.asarray):
    numpy_max = np.max(x, axis=-1)
    mygrad_max = no_autodiff(amax, to_numpy=True)(x, axis=-1)
    assert isinstance(mygrad_max, np.ndarray)
    assert_array_equal(numpy_max, mygrad_max)


def test_nested_context():
    assert _tracking.TRACK_GRAPH, "before all"

    with no_autodiff:
        assert not _tracking.TRACK_GRAPH, "level 1, pre-2"

        with no_autodiff:
            assert not _tracking.TRACK_GRAPH, "level 2"

        assert not _tracking.TRACK_GRAPH, "level 1, post-2"

    assert _tracking.TRACK_GRAPH, "post all"


def compose(iter_of_funcs):
    """(..., f, g, h) -> ... ∘ f ∘ g ∘ h"""
    funcs = list(iter_of_funcs)
    f = funcs.pop()
    for g in funcs[::-1]:
        f = g(f)
    return f


@given(depth=st.integers(1, 10))
def test_graph_tracking_through_multiple_context_depths(depth: int):
    depth_tracker = dict(cnt=0)

    def check_graph_tracking(func):
        @no_autodiff
        def wrapper(*args, **kwargs):
            depth_tracker["cnt"] += 1
            assert (
                _tracking.TRACK_GRAPH is False
            ), f"(before) depth: {depth_tracker['cnt']}"
            out = func(*args, **kwargs)
            assert (
                _tracking.TRACK_GRAPH is False
            ), f"(after) depth: {depth_tracker['cnt']}"
            return out

        return wrapper

    def func(x):
        assert _tracking.TRACK_GRAPH is False
        return +x

    f = compose(check_graph_tracking for _ in range(depth))(func)
    f(Tensor([1.0]))
