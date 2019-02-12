from mygrad._utils import reduce_broadcast
from tests.custom_strategies import broadcastable_shape
import numpy as np

from numpy.testing import assert_allclose

from pytest import raises
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


def test_bad_gradient_dimensionality():
    """ test that grad.dim < len(var_shape) raises ValueError"""
    var_shape = (1, 2, 3)
    grad = np.empty((1, 2))
    with raises(ValueError):
        reduce_broadcast(grad=grad, var_shape=var_shape)


@given(grad=hnp.arrays(dtype=float, shape=hnp.array_shapes(), elements=st.floats(-100, 100)))
def test_broadcast_scalar(grad):
    """ test when grad was broadcasted from a scalar"""
    assert_allclose(reduce_broadcast(grad, tuple()), grad.sum())


@given(grad=hnp.arrays(dtype=float, shape=hnp.array_shapes(), elements=st.floats(-100, 100)))
def test_reduce_broadcast_same_shape(grad):
    """ test when no broadcasting occurred"""
    var_shape=grad.shape
    reduced_grad = reduce_broadcast(grad=grad, var_shape=var_shape)
    assert_allclose(actual=reduced_grad, desired=grad)


@given(var_shape=hnp.array_shapes(min_side=2),
       data=st.data())
def test_reduce_broadcast_nokeepdim(var_shape, data):
    """ example broadcasting: (2, 3) -> (5, 2, 3)"""
    grad_shape = data.draw(broadcastable_shape(shape=var_shape,
                                               min_dim=len(var_shape) + 1,
                                               max_dim=len(var_shape) + 3,
                                               allow_singleton=False),
                           label="grad_shape")
    grad = np.ones(grad_shape, dtype=float)

    reduced_grad = reduce_broadcast(grad=grad, var_shape=var_shape)
    reduced_grad *= np.prod(var_shape) / grad.size  # scale reduced-grad so all elements are 1
    assert_allclose(actual=reduced_grad, desired=np.ones(var_shape))


@given(var_shape=hnp.array_shapes(),
       data=st.data())
def test_reduce_broadcast_keepdim(var_shape, data):
    """ example broadcasting: (2, 1, 4) -> (2, 5, 4)"""
    grad = data.draw(hnp.arrays(dtype=float,
                     shape=broadcastable_shape(shape=var_shape,
                                               min_dim=len(var_shape),
                                               max_dim=len(var_shape)),
                                               elements=st.just(1.)),
                     label='grad')

    reduced_grad = reduce_broadcast(grad=grad, var_shape=var_shape)
    assert reduced_grad.shape == tuple(i if i < j else j for i, j in zip(var_shape, grad.shape))
    assert (i == 1 for i, j in zip(var_shape, grad.shape) if i < j)
    sum_axes = tuple(n for n, (i, j) in enumerate(zip(var_shape, grad.shape)) if i != j)
    assert_allclose(actual=reduced_grad, desired=grad.sum(axis=sum_axes, keepdims=True))


@given(grad=hnp.arrays(dtype=float, shape=(5, 3, 4, 2), elements=st.floats(-.01, .01)))
def test_hybrid_broadcasting(grad):
    """ tests new-dim and keep-dim broadcasting
         (3, 1, 2) -> (5, 3, 4, 2)"""
    var_shape = (3, 1, 2)
    reduced = reduce_broadcast(grad=grad, var_shape=var_shape)
    answer = grad.sum(axis=0).sum(axis=-2, keepdims=True)
    assert_allclose(actual=reduced, desired=answer)





