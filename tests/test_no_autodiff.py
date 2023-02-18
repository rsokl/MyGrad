import inspect

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note
from numpy.testing import assert_array_equal

import mygrad._utils.graph_tracking as _tracking
from mygrad import Tensor, amax, no_autodiff
from mygrad.nnet.activations import soft_sign
from tests.custom_strategies import tensors


@pytest.mark.usefixtures("seal_graph_tracking")
def test_graph_tracking_is_on_by_default():
    assert _tracking.TRACK_GRAPH is True


def test_no_autodiff_doesnt_restrict_tensor_type():
    with no_autodiff:
        x = Tensor(1.0, dtype=np.complex64)

    # non-floats should be constant
    assert x.constant is True


@pytest.mark.usefixtures("seal_graph_tracking")
@given(x=tensors(shape=(1,), elements=st.floats(-100, 100)), constant=st.booleans())
def test_no_autodiff_context_manager(x: Tensor, constant: bool):
    with no_autodiff:
        # test soft_sign so that we pass through multi-node
        # graph
        y = soft_sign(x, constant=constant)

        assert y.constant is constant
        assert y.creator is None
        assert y.data.flags.writeable

        assert not x._ops
        assert x.data.flags.writeable

        y.backward()  # should be no-op

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


@pytest.mark.usefixtures("seal_graph_tracking")
@given(tensors())
def test_no_track_view_children(x: Tensor):
    with no_autodiff:
        _ = x[...]

    assert not x._view_children


@pytest.mark.usefixtures("seal_graph_tracking")
@given(old_x=tensors(read_only=st.booleans()))
def test_no_autodiff_does_not_unlock_memory(old_x: Tensor):
    x = +old_x

    with pytest.raises(ValueError):  # data is read-only
        with no_autodiff:
            x[...] = -1


@pytest.mark.usefixtures("seal_graph_tracking")
@given(old_x=tensors())
def test_no_autodiff_on_in_place_op_does_not_track_graph(old_x: Tensor):
    with no_autodiff:
        x = old_x[...]
        x[...] = 0

    assert_array_equal(x, np.zeros_like(x))
    assert_array_equal(old_x, np.zeros_like(x))
    assert old_x.data.flags.writeable is True
    assert x.data.flags.writeable is True

    assert x.constant is False  # constant doesn't propagate through graph

    assert x.base is None
    assert x.creator is None
    assert not old_x._ops
    assert not old_x._view_children
    if x.size:
        assert np.shares_memory(x, old_x)


@pytest.mark.usefixtures("seal_graph_tracking")
def test_no_autodiff_context_manager_restores_state_via_finally_clause():
    with pytest.raises(ValueError):
        with no_autodiff:
            assert not _tracking.TRACK_GRAPH
            raise ValueError()  # we should restore graph track despite raise

    assert _tracking.TRACK_GRAPH is True


@pytest.mark.usefixtures("seal_graph_tracking")
@given(x=tensors(shape=(1,), elements=st.floats(-100, 100)), constant=st.booleans())
def test_no_autodiff_decorator(x: Tensor, constant: bool):
    @no_autodiff
    def func(x, constant=False):
        return soft_sign(x, constant=constant)

    y = func(x, constant=constant)

    assert y.constant is constant
    assert y.creator is None
    assert not y._ops
    assert y.data.flags.writeable

    assert not x._ops
    assert x.data.flags.writeable


@pytest.mark.usefixtures("seal_graph_tracking")
def test_no_autodiff_decorator_restores_state_via_finally():
    @no_autodiff
    def func():
        assert not _tracking.TRACK_GRAPH
        raise ValueError()  # state should be restored despite error

    with pytest.raises(ValueError):
        func()

    assert _tracking.TRACK_GRAPH is True


@pytest.mark.usefixtures("seal_graph_tracking")
def test_decorator_is_transparent_to_function_information():
    def undecorated(x, y, z=None, *, kwarg, **kwargs):
        """a very special docstring"""

    decorated = no_autodiff(undecorated)

    assert inspect.signature(undecorated) == inspect.signature(decorated)
    assert undecorated.__doc__ == decorated.__doc__
    assert undecorated.__name__ == decorated.__name__


@pytest.mark.usefixtures("seal_graph_tracking")
@given(tensors(shape=(3, 2, 4), elements=st.floats(-100, 100)).map(np.asarray))
def test_to_numpy_returns_numpy_array(x: np.asarray):
    numpy_max = np.max(x, axis=-1)
    mygrad_max = no_autodiff(amax, to_numpy=True)(x, axis=-1)
    assert isinstance(mygrad_max, np.ndarray)
    assert_array_equal(numpy_max, mygrad_max)


@pytest.mark.usefixtures("seal_graph_tracking")
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


@pytest.mark.usefixtures("seal_graph_tracking")
@given(depth=st.integers(1, 10))
def test_graph_tracking_through_multiple_context_depths(depth: int):
    depth_tracker = dict(cnt=0)

    def check_graph_tracking(func):
        def wrapper(*args, **kwargs):
            depth_tracker["cnt"] += 1
            note(f"depth: {depth_tracker}")
            track_state = _tracking.TRACK_GRAPH

            out = no_autodiff(func)(*args, **kwargs)

            assert (
                _tracking.TRACK_GRAPH is track_state
            ), f"(after) depth: {depth_tracker['cnt']}"

            return out

        return wrapper

    def func(x):
        assert _tracking.TRACK_GRAPH is False
        return +x

    f = compose([check_graph_tracking] * depth + [func])

    assert _tracking.TRACK_GRAPH is True
    f(Tensor([1.0]))
    assert _tracking.TRACK_GRAPH is True
