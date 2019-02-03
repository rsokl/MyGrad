""" Test `numerical_gradient`, `numerical_derivative`, and `broadcast_check`"""

from tests.utils.numerical_gradient import numerical_gradient, numerical_gradient_full

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given, settings

from numpy.testing import assert_allclose


def unary_func(x): return x ** 2


def binary_func(x, y): return x * y ** 2


def ternary_func(x, y, z): return z * x * y ** 2


@given(st.data())
def test_numerical_gradient_no_broadcast(data):

    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    y = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    z = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))


    # check variable-selection
    assert numerical_gradient(unary_func, x, back_grad=grad, vary_ind=[])[0] is None

    # no broadcast
    dx, = numerical_gradient(unary_func, x, back_grad=grad)

    assert_allclose(dx, grad * 2 * x)

    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    assert_allclose(dx, grad * y ** 2)
    assert_allclose(dy, grad * 2 * x * y)

    dx, dy, dz = numerical_gradient(ternary_func, x, y, z, back_grad=grad)
    assert_allclose(dx, grad * z * y ** 2)
    assert_allclose(dy, grad * z * 2 * x * y)
    assert_allclose(dz, grad * x * y ** 2)


@given(st.data())
def test_numerical_gradient_x_broadcast(data):

    x = data.draw(hnp.arrays(shape=(3, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    y = data.draw(hnp.arrays(shape=(2, 3, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=(2, 3, 4),
                                dtype=float,
                                elements=st.floats(-100, 100)))

    # broadcast x
    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    assert_allclose(dx, (grad * y ** 2).sum(axis=0))
    assert_allclose(dy, grad * 2 * x * y)


@given(st.data())
def test_numerical_gradient_y_broadcast(data):

    y = data.draw(hnp.arrays(shape=(3, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    x = data.draw(hnp.arrays(shape=(2, 3, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=(2, 3, 4),
                                dtype=float,
                                elements=st.floats(-100, 100)))

    # broadcast x
    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    assert_allclose(dx, grad * y ** 2)
    assert_allclose(dy, (grad * 2 * x * y).sum(axis=0))


@given(st.data())
def test_numerical_gradient_xy_broadcast(data):

    x = data.draw(hnp.arrays(shape=(2, 1, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    y = data.draw(hnp.arrays(shape=(1, 3, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=(2, 3, 4),
                                dtype=float,
                                elements=st.floats(-100, 100)))

    # broadcast x
    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    x_grad = (grad * y ** 2).sum(axis=1, keepdims=True)
    y_grad = (grad * 2 * x * y).sum(axis=0, keepdims=True)
    assert_allclose(dx, x_grad, atol=1e-5, rtol=1e-5)
    assert_allclose(dy, y_grad, atol=1e-5, rtol=1e-5)


@given(st.data())
def test_numerical_gradient_full_xy_broadcast(data):

    x = data.draw(hnp.arrays(shape=(2, 1, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    y = data.draw(hnp.arrays(shape=(1, 3, 4),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=(2, 3, 4),
                                dtype=float,
                                elements=st.floats(-100, 100)))

    # broadcast x
    dx, dy = numerical_gradient_full(binary_func, x, y, back_grad=grad)
    x_grad = (grad * y ** 2).sum(axis=1, keepdims=True)
    y_grad = (grad * 2 * x * y).sum(axis=0, keepdims=True)
    assert_allclose(dx, x_grad, atol=1e-2, rtol=1e-2)
    assert_allclose(dy, y_grad, atol=1e-2, rtol=1e-2)
