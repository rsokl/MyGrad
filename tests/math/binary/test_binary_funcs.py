""" Test all binary arithmetic operations, checks for appropriate broadcast behavior"""
from ...wrappers.uber import fwdprop_test_factory, backprop_test_factory

from mygrad import add, subtract, multiply, divide, power, logaddexp
from mygrad import logaddexp2, maximum, minimum

import numpy as np


from mygrad.tensor_base import Tensor
from numpy.testing import assert_allclose

from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from ...utils.numerical_gradient import numerical_gradient_full
from ...custom_strategies import broadcastable_shape


@fwdprop_test_factory(mygrad_func=add, true_func=np.add, num_arrays=2)
def test_add_fwd(): pass


@backprop_test_factory(mygrad_func=add, true_func=np.add, num_arrays=2,
                       atol=1e-4, rtol=1e-4)
def test_add_bkwd(): pass


@fwdprop_test_factory(mygrad_func=subtract, true_func=np.subtract, num_arrays=2)
def test_subtract_fwd(): pass


@backprop_test_factory(mygrad_func=subtract, true_func=np.subtract, num_arrays=2,
                       atol=1e-4, rtol=1e-4)
def test_subtract_bkwd(): pass


@fwdprop_test_factory(mygrad_func=multiply, true_func=np.multiply, num_arrays=2)
def test_multiply_fwd(): pass


@backprop_test_factory(mygrad_func=multiply, true_func=np.multiply, atol=1e-4, rtol=1e-4, num_arrays=2)
def test_multiply_bkwd(): pass


@fwdprop_test_factory(mygrad_func=divide, true_func=np.divide, index_to_bnds={1: (1, 10)}, num_arrays=2)
def test_divide_fwd(): pass


@backprop_test_factory(mygrad_func=divide, true_func=np.divide, index_to_bnds={1: (1, 10)}, num_arrays=2,
                       atol=1e-4, rtol=1e-4)
def test_divide_bkwd(): pass


@fwdprop_test_factory(mygrad_func=power, true_func=np.power,
                      index_to_bnds={0: (1, 10), 1: (-3, 3)},
                      num_arrays=2)
def test_power_fwd(): pass


@backprop_test_factory(mygrad_func=power, true_func=np.power,
                       index_to_bnds={0: (1, 10), 1: (-3, 3)},
                       num_arrays=2, atol=1e-4, rtol=1e-4)
def test_power_bkwd(): pass


@fwdprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp, num_arrays=2)
def test_logaddexp_fwd(): pass


@backprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp, num_arrays=2,
                       as_decimal=False, atol=1e-4, rtol=1e-4)
def test_logaddexp_bkwd(): pass


@fwdprop_test_factory(mygrad_func=logaddexp2, true_func=np.logaddexp2, num_arrays=2)
def test_logaddexp2_fwd(): pass


@backprop_test_factory(mygrad_func=logaddexp2, true_func=np.logaddexp2, num_arrays=2,
                       as_decimal=False, atol=1e-4, rtol=1e-4)
def test_logaddexp2_bkwd(): pass


@fwdprop_test_factory(mygrad_func=maximum, true_func=np.maximum, num_arrays=2)
def test_maximum_fwd(): pass


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_maximum_bkwd(x, data):
    y = data.draw(hnp.arrays(shape=broadcastable_shape(x.shape, max_dim=5),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    assume(not np.any(np.isclose(x, y)))

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    o = maximum(x_arr, y_arr)

    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")
    (o * grad).sum().backward()


    dx, dy = numerical_gradient_full(np.maximum, x, y, back_grad=grad,
                                     as_decimal=True)

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


def test_maximum_minimum_bkwd_equal():
    """ regression test for documented behavior of maximum/minimum where
        x == y"""

    x = Tensor([1., 0., 2.])
    y = Tensor([2., 0., 1.])

    o = maximum(x, y)
    o.backward()

    assert_allclose(x.grad, [0., 0., 1])
    assert_allclose(y.grad, [1., 0., 0])
    o.null_gradients()

    x = Tensor([1., 0., 2.])
    y = Tensor([2., 0., 1.])

    o = minimum(x, y)
    o.backward()

    assert_allclose(x.grad, [1., 0., 0.])
    assert_allclose(y.grad, [0., 0., 1.])
    o.null_gradients()


@fwdprop_test_factory(mygrad_func=minimum, true_func=np.minimum, num_arrays=2)
def test_minimum_fwd(): pass


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_minimum_bkwd(x, data):
    """ index conforms strictly to basic indexing """

    y = data.draw(hnp.arrays(shape=broadcastable_shape(x.shape, max_dim=5),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    assume(not np.any(np.isclose(x, y)))

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    o = minimum(x_arr, y_arr)

    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")
    (o * grad).sum().backward()


    dx, dy = numerical_gradient_full(np.minimum, x, y, back_grad=grad,
                                     as_decimal=True)

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)
