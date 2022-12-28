from typing import Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose
from pytest import raises

from mygrad._utils import WeakRefIterable, reduce_broadcast
from tests.custom_strategies import broadcastable_shapes


def test_weakrefs_list():
    class A:
        pass

    a = A()
    b = A()
    c = A()
    items = WeakRefIterable([a, b, c])
    assert len(items) == 3
    del b  # weakref to b returns None now
    a_val, c_val = items

    assert a_val is a
    assert c_val is c


def test_weakrefs_bool():
    class A:
        pass

    a = A()
    b = A()
    c = A()
    assert bool(WeakRefIterable([a, b, c])) is True
    assert bool(WeakRefIterable([])) is False

    weak_list = WeakRefIterable([a])
    assert bool(weak_list) is True
    del a
    assert bool(weak_list) is False


@given(shapes=hnp.mutually_broadcastable_shapes(num_shapes=2, max_dims=5))
def test_reduce_broadcast_shape_consistency(shapes: hnp.BroadcastableShapes):
    grad = np.zeros(shapes.result_shape)

    assert (
        reduce_broadcast(grad, var_shape=shapes.input_shapes[0]).shape
        == shapes.input_shapes[0]
    )
    assert (
        reduce_broadcast(grad, var_shape=shapes.input_shapes[1]).shape
        == shapes.input_shapes[1]
    )


@given(
    shapes=hnp.array_shapes(min_dims=1, max_dims=10).flatmap(
        lambda shape: st.tuples(
            st.just(shape), hnp.array_shapes(min_dims=0, max_dims=len(shape) - 1)
        )
    )
)
def test_bad_gradient_dimensionality(shapes: Tuple[Tuple[int, ...], Tuple[int, ...]]):
    """test that grad.dim < len(var_shape) raises ValueError"""
    var_shape = shapes[0]
    grad = np.empty(shapes[1])
    with raises(ValueError):
        reduce_broadcast(grad=grad, var_shape=var_shape)


@given(
    grad=hnp.arrays(
        dtype=float, shape=hnp.array_shapes(), elements=st.floats(-100, 100)
    )
)
def test_broadcast_scalar(grad):
    """test when grad was broadcasted from a scalar"""
    assert_allclose(reduce_broadcast(grad, tuple()), grad.sum())


@given(
    grad=hnp.arrays(
        dtype=float, shape=hnp.array_shapes(), elements=st.floats(-100, 100)
    )
)
def test_reduce_broadcast_same_shape(grad):
    """test when no broadcasting occurred"""
    var_shape = grad.shape
    reduced_grad = reduce_broadcast(grad=grad, var_shape=var_shape)
    assert_allclose(actual=reduced_grad, desired=grad)


@given(var_shape=hnp.array_shapes(min_side=2), data=st.data())
def test_reduce_broadcast_nokeepdim(var_shape, data):
    """example broadcasting: (2, 3) -> (5, 2, 3)"""
    grad_shape = data.draw(
        broadcastable_shapes(
            shape=var_shape,
            min_dims=len(var_shape) + 1,
            max_dims=len(var_shape) + 3,
            min_side=2,
        ),
        label="grad_shape",
    )
    grad = np.ones(grad_shape, dtype=float)

    reduced_grad = reduce_broadcast(grad=grad, var_shape=var_shape)
    reduced_grad *= (
        np.prod(var_shape) / grad.size
    )  # scale reduced-grad so all elements are 1
    assert_allclose(actual=reduced_grad, desired=np.ones(var_shape))


@given(var_shape=hnp.array_shapes(), data=st.data())
def test_reduce_broadcast_keepdim(var_shape, data):
    """example broadcasting: (2, 1, 4) -> (2, 5, 4)"""
    grad = data.draw(
        hnp.arrays(
            dtype=float,
            shape=broadcastable_shapes(
                shape=var_shape, min_dims=len(var_shape), max_dims=len(var_shape)
            ),
            elements=st.just(1.0),
        ),
        label="grad",
    )

    reduced_grad = reduce_broadcast(grad=grad, var_shape=var_shape)
    assert reduced_grad.shape == tuple(
        i if i < j else j for i, j in zip(var_shape, grad.shape)
    )
    assert (i == 1 for i, j in zip(var_shape, grad.shape) if i < j)
    sum_axes = tuple(n for n, (i, j) in enumerate(zip(var_shape, grad.shape)) if i != j)
    assert_allclose(actual=reduced_grad, desired=grad.sum(axis=sum_axes, keepdims=True))


@given(
    grad=hnp.arrays(dtype=float, shape=(5, 3, 4, 2), elements=st.floats(-0.01, 0.01))
)
def test_hybrid_broadcasting(grad):
    """tests new-dim and keep-dim broadcasting
    (3, 1, 2) -> (5, 3, 4, 2)"""
    var_shape = (3, 1, 2)
    reduced = reduce_broadcast(grad=grad, var_shape=var_shape)
    answer = grad.sum(axis=0).sum(axis=-2, keepdims=True)
    assert_allclose(actual=reduced, desired=answer)
