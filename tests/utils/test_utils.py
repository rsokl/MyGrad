""" Test `numerical_gradient`, `numerical_derivative`, and `broadcast_check`"""

from tests.utils.numerical_gradient import numerical_gradient, numerical_gradient_full

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given

from tests.custom_strategies import valid_axes
import numpy as np
from numpy.testing import assert_allclose


def unary_func(x): return x ** 2


def binary_func(x, y): return x * y ** 2


def ternary_func(x, y, z): return z * x * y ** 2


@given(data=st.data(),
       x=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       as_decimal=st.booleans())
def test_numerical_gradient_no_broadcast(data, x, as_decimal):
    atol, rtol = (1e-7, 1e-7) if as_decimal else (1e-2, 1e-2)
    y = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)),
                  label="y")

    z = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)),
                  label="z")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)),
                     label="grad")


    # check variable-selection
    assert numerical_gradient(unary_func, x, back_grad=grad,
                              vary_ind=[], as_decimal=as_decimal)[0] is None

    # no broadcast
    dx, = numerical_gradient(unary_func, x, back_grad=grad, as_decimal=as_decimal)

    assert_allclose(dx, grad * 2 * x, atol=atol, rtol=rtol)

    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    assert_allclose(dx, grad * y ** 2, atol=atol, rtol=rtol)
    assert_allclose(dy, grad * 2 * x * y, atol=atol, rtol=rtol)

    dx, dy, dz = numerical_gradient(ternary_func, x, y, z, back_grad=grad)
    assert_allclose(dx, grad * z * y ** 2, atol=atol, rtol=rtol)
    assert_allclose(dy, grad * z * 2 * x * y, atol=atol, rtol=rtol)
    assert_allclose(dz, grad * x * y ** 2, atol=atol, rtol=rtol)


@given(data=st.data(),
       x=hnp.arrays(shape=(3, 4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       y=hnp.arrays(shape=(2, 3, 4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       grad=hnp.arrays(shape=(2, 3, 4),
                       dtype=float,
                       elements=st.floats(-100, 100)),
       as_decimal=st.booleans())
def test_numerical_gradient_x_broadcast(data, x, y, grad, as_decimal):
    atol, rtol = (1e-7, 1e-7) if as_decimal else (1e-2, 1e-2)

    # broadcast x
    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    assert_allclose(dx, (grad * y ** 2).sum(axis=0), atol=atol, rtol=rtol)
    assert_allclose(dy, grad * 2 * x * y, atol=atol, rtol=rtol)


@given(data=st.data(),
       y=hnp.arrays(shape=(3, 4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       x=hnp.arrays(shape=(2, 3, 4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       grad=hnp.arrays(shape=(2, 3, 4),
                       dtype=float,
                       elements=st.floats(-100, 100)),
       as_decimal=st.booleans())
def test_numerical_gradient_y_broadcast(data, x, y, grad, as_decimal):
    atol, rtol = (1e-7, 1e-7) if as_decimal else (1e-2, 1e-2)

    # broadcast x
    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    assert_allclose(dx, grad * y ** 2, atol=atol, rtol=rtol)
    assert_allclose(dy, (grad * 2 * x * y).sum(axis=0), atol=atol, rtol=rtol)


@given(data=st.data(),
       x=hnp.arrays(shape=(2, 1, 4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       y=hnp.arrays(shape=(1, 3, 4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       grad=hnp.arrays(shape=(2, 3, 4),
                       dtype=float,
                       elements=st.floats(-100, 100)),
       as_decimal=st.booleans())
def test_numerical_gradient_xy_broadcast(data, x, y, grad, as_decimal):
    atol, rtol = (1e-7, 1e-7) if as_decimal else (1e-2, 1e-2)

    # broadcast x
    dx, dy = numerical_gradient(binary_func, x, y, back_grad=grad)
    x_grad = (grad * y ** 2).sum(axis=1, keepdims=True)
    y_grad = (grad * 2 * x * y).sum(axis=0, keepdims=True)
    assert_allclose(dx, x_grad, atol=atol, rtol=rtol)
    assert_allclose(dy, y_grad, atol=atol, rtol=rtol)


@given(x=hnp.arrays(dtype=float,
                    elements=st.floats(-1, 1),
                    shape=hnp.array_shapes(max_side=3)),
       grad=st.floats(-1, 1),
       keepdims=st.booleans(),
       as_decimal=st.booleans(),
       data=st.data())
def test_numerical_gradient_vary_each(x, grad, keepdims, as_decimal, data):
    atol, rtol = (1e-7, 1e-7) if as_decimal else (1e-2, 1e-2)
    axes = data.draw(valid_axes(ndim=x.ndim), label="axes")
    kwargs = dict(axis=axes, keepdims=keepdims)
    dx, = numerical_gradient_full(np.sum, x, back_grad=np.array(grad), kwargs=kwargs,
                                  as_decimal=as_decimal)
    x_grad = np.full_like(x, fill_value=grad)
    assert_allclose(actual=dx, desired=x_grad, atol=atol, rtol=rtol)
