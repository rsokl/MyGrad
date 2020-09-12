import inspect

import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_array_equal

import mygrad._graph_tracking as _tracking
from mygrad import Tensor, amax, no_autodiff
from tests.custom_strategies import tensors


def test_graph_tracking_is_on_by_default():
    assert _tracking.TRACK_GRAPH is True


@given(x=tensors(shape=(1,)))
def test_no_autodiff_context_manager(x: Tensor):

    with no_autodiff:
        y = +x

    assert y.constant
    assert y.creator is None
    assert not y._ops
    assert x.data.flags.writeable

    assert not x._ops
    assert x.data.flags.writeable


@given(old_x=tensors(shape=(2,), constant=False))
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
    assert x.data.flags.writeable

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


@given(tensors(shape=(3, 2, 4)).map(np.asarray))
def test_to_numpy_returns_numpy_array(x: np.asarray):
    numpy_max = np.max(x, axis=-1)
    mygrad_max = no_autodiff(amax, to_numpy=True)(x, axis=-1)
    assert isinstance(mygrad_max, np.ndarray)
    assert_array_equal(numpy_max, mygrad_max)
