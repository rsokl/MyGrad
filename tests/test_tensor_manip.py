from mygrad.tensor_base import Tensor
from mygrad import transpose, moveaxis, swapaxes, squeeze
from numpy.testing import assert_allclose
import numpy as np

from .custom_strategies import valid_axes

import hypothesis.extra.numpy as hnp
from hypothesis import given, assume
import hypothesis.strategies as st
from .utils.numerical_gradient import numerical_gradient_full


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_moveaxis(x, data):
    src = data.draw(valid_axes(x.ndim, permit_none=False), label="source")
    dest = data.draw(valid_axes(x.ndim, permit_none=False), label="destination")
    assume(len(src) == len(dest))

    x_arr = Tensor(np.copy(x))

    o = moveaxis(x_arr, src, dest, constant=False)
    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

    o.backward(grad)

    def f(x): return np.moveaxis(x, src, dest)

    assert_allclose(o.data, f(x))

    dx, = numerical_gradient_full(f, x, back_grad=grad, as_decimal=True)

    assert_allclose(x_arr.grad, dx)


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_swapaxes(x, data):
    axis1 = data.draw(st.integers(-x.ndim, x.ndim - 1), label="axis1")
    axis2 = data.draw(st.integers(-x.ndim, x.ndim - 1), label="axis2")

    x_arr = Tensor(np.copy(x))

    o = swapaxes(x_arr, axis1, axis2, constant=False)
    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

    o.backward(grad)

    def f(x): return np.swapaxes(x, axis1, axis2)

    assert_allclose(o.data, f(x))

    dx, = numerical_gradient_full(f, x, back_grad=grad, as_decimal=True)

    assert_allclose(x_arr.grad, dx)
    
    
@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_transpose(x, data):
    axes = data.draw(valid_axes(x.ndim), label="axes")
    if axes is not None:
        assume(len(axes) == x.ndim)

    x_arr = Tensor(np.copy(x))

    o = transpose(x_arr, axes, constant=False)
    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

    o.backward(grad)

    def f(x): return np.transpose(x, axes)

    assert_allclose(o.data, f(x))

    dx, = numerical_gradient_full(f, x, back_grad=grad, as_decimal=True)

    assert_allclose(x_arr.grad, dx)


def test_transpose_property():
    dat = np.arange(6).reshape(2, 3)
    x = Tensor(dat)
    f = x.T
    f.backward(dat.T)

    assert_allclose(f.data, dat.T)
    assert_allclose(x.grad, dat)


def test_transpose_method():
    dat = np.arange(24).reshape(2, 3, 4)
    x = Tensor(dat)

    # passing tuple of integers
    f = x.transpose((2, 1, 0))
    f.backward(dat.transpose((2, 1, 0)))

    assert_allclose(f.data, dat.transpose((2, 1, 0)))
    assert_allclose(x.grad, dat)
    
    # passing integers directly
    x = Tensor(dat)
    f = x.transpose(2, 1, 0)
    f.backward(dat.transpose((2, 1, 0)))

    assert_allclose(f.data, dat.transpose((2, 1, 0)))
    assert_allclose(x.grad, dat)

    # passing integers directly
    x = Tensor(dat)
    f = x.transpose()
    f.backward(dat.transpose())

    assert_allclose(f.data, dat.transpose())
    assert_allclose(x.grad, dat)