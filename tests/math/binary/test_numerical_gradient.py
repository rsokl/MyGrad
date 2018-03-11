""" Test `numerical_gradient`"""
from .binary_func import numerical_gradient

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_numerical_gradient_no_broadcast(data):
    def f(x, y): return x * y**2

    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))

    y = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    # no broadcast
    dx, dy = numerical_gradient(f, x=x, y=y, back_grad=grad)
    assert np.allclose(dx, grad * y ** 2)
    assert np.allclose(dy, grad * 2 * x * y)


@given(st.data())
def test_numerical_gradient_x_broadcast(data):
    def f(x, y): return x * y**2

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
    dx, dy = numerical_gradient(f, x=x, y=y, back_grad=grad)
    assert np.allclose(dx, (grad * y ** 2).sum(axis=0))
    assert np.allclose(dy, grad * 2 * x * y)


@given(st.data())
def test_numerical_gradient_y_broadcast(data):
    def f(x, y): return x * y**2

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
    dx, dy = numerical_gradient(f, x=x, y=y, back_grad=grad)
    assert np.allclose(dx, grad * y ** 2)
    assert np.allclose(dy, (grad * 2 * x * y).sum(axis=0))


@given(st.data())
def test_numerical_gradient_xy_broadcast(data):
    def f(x, y): return x * y**2

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
    dx, dy = numerical_gradient(f, x=x, y=y, back_grad=grad)
    x_grad = (grad * y ** 2).sum(axis=1, keepdims=True)
    y_grad = (grad * 2 * x * y).sum(axis=0, keepdims=True)
    assert np.allclose(dx, x_grad)
    assert np.allclose(dy, y_grad)
