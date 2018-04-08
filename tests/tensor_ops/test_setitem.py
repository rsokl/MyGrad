from mygrad.tensor_base import Tensor
import numpy as np
from numpy.testing import assert_allclose

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from ..utils.numerical_gradient import numerical_gradient_full
from ..custom_strategies import basic_index, adv_integer_index, broadcastable_shape


def setitem(x, y, index):
    x_copy = np.copy(x)
    x_copy[index] = y
    return x_copy


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_basic_index(x, data):
    index = data.draw(basic_index(x.shape), label="index")
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

    x0 = np.copy(x)
    y0 = np.copy(y)

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    x0[index] = y0
    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_adv_int_index(x, data):
    index = data.draw(adv_integer_index(x.shape), label="index")
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

    x0 = np.copy(x)
    y0 = np.copy(y)

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    x0[index] = y0
    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)

