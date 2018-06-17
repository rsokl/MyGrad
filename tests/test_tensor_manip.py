from itertools import permutations
from mygrad.tensor_base import Tensor
from mygrad import transpose, moveaxis, swapaxes, squeeze
from numpy.testing import assert_allclose
import numpy as np

from .custom_strategies import valid_axes
from .utils.numerical_gradient import numerical_gradient_full

import hypothesis.extra.numpy as hnp
from hypothesis import given, assume
import hypothesis.strategies as st

from pytest import raises


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

    out = transpose(x, constant=True)
    assert out.constant and not x_arr.constant


def test_transpose_property():
    dat = np.arange(6).reshape(2, 3)
    x = Tensor(dat)
    f = x.T
    f.backward(dat.T)

    assert_allclose(f.data, dat.T)
    assert_allclose(x.grad, dat)


def test_transpose_method():
    dat = np.arange(24).reshape(2, 3, 4)

    for axes in permutations(range(3)):
        # passing tuple of integers
        x = Tensor(dat)
        f = x.transpose(axes)
        f.backward(dat.transpose(axes))

        assert_allclose(f.data, dat.transpose(axes))
        assert_allclose(x.grad, dat)

        # passing integers directly
        x = Tensor(dat)
        f = x.transpose(*axes)
        f.backward(dat.transpose(axes))

        assert_allclose(f.data, dat.transpose(axes), err_msg="{}".format(axes))
        assert_allclose(x.grad, dat, err_msg="{}".format(axes))

    # passing integers directly
    x = Tensor(dat)
    f = x.transpose()
    f.backward(dat.transpose())

    assert_allclose(f.data, dat.transpose())
    assert_allclose(x.grad, dat)

    # check that constant=True works
    x = Tensor(dat)
    f = x.transpose(constant=True)
    assert f.constant and not x.constant

    f = x.transpose(1, 0, 2, constant=True)
    assert f.constant and not x.constant


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=2, max_dims=3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_squeeze(x, data):
    axes = data.draw(valid_axes(x.ndim), label="axes")
    x_arr = Tensor(np.copy(x))
    x_arr2 = Tensor(np.copy(x))

    def f(x): return np.squeeze(x, axes)

    try:
        numpy_out = np.squeeze(x, axes)
    except ValueError as e:
        with raises(ValueError):
            squeeze(x_arr, axes, constant=False)
        return

    o = squeeze(x_arr, axes, constant=False)
    o_method = x_arr2.squeeze(axes)
    assert_allclose(o.data, numpy_out)
    assert_allclose(o_method.data, numpy_out)

    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")
    o.backward(grad)
    o_method.backward(grad)

    dx, = numerical_gradient_full(f, x, back_grad=grad, as_decimal=True)

    assert_allclose(x_arr.grad, dx)
    assert_allclose(x_arr2.grad, dx)
