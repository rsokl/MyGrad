""" Test all binary arithmetic operations, checks for appropriate broadcast behavior"""
from ...wrappers.binary_func import fwdprop_test_factory, backprop_test_factory

from mygrad.math import add, subtract, multiply, divide, power, logaddexp
import numpy as np
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from ...custom_strategies import broadcastable_shape
import mygrad.math as mg
from mygrad import Tensor


@fwdprop_test_factory(mygrad_func=add, true_func=np.add)
def test_add_fwd(): pass


@backprop_test_factory(mygrad_func=add, true_func=np.add)
def test_add_bkwd(): pass


@fwdprop_test_factory(mygrad_func=subtract, true_func=np.subtract)
def test_subtract_fwd(): pass


@backprop_test_factory(mygrad_func=subtract, true_func=np.subtract)
def test_subtract_bkwd(): pass


@fwdprop_test_factory(mygrad_func=multiply, true_func=np.multiply)
def test_multiply_fwd(): pass


@backprop_test_factory(mygrad_func=multiply, true_func=np.multiply)
def test_multiply_bkwd(): pass


@fwdprop_test_factory(mygrad_func=divide, true_func=np.divide, ybnds=(1, 10))
def test_divide_fwd(): pass


@backprop_test_factory(mygrad_func=divide, true_func=np.divide, ybnds=(1, 10))
def test_divide_bkwd(): pass


@fwdprop_test_factory(mygrad_func=power, true_func=np.power, xbnds=(1, 10), ybnds=(-3, 3))
def test_power_fwd(): pass


@backprop_test_factory(mygrad_func=power, true_func=np.power, xbnds=(1, 10), ybnds=(-3, 3))
def test_power_bkwd(): pass


@fwdprop_test_factory(mygrad_func=add, true_func=np.add)
def test_add_fwd(): pass


@backprop_test_factory(mygrad_func=add, true_func=np.add)
def test_add_bkwd(): pass


@fwdprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp)
def test_logaddexp_fwd(): pass


# built-in numpy logaddexp doesn't work with object arrays
@given(data=st.data(),
       x=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                    dtype=float,
                    elements=st.floats(-10, 10)))
def test_logaddexp_bkwd(data, x):
    """ Performs hypothesis unit test for checking back-propagation
        through a `mygrad` op.

        Raises
        ------
        AssertionError"""

    y = data.draw(hnp.arrays(shape=broadcastable_shape(x.shape),
                             dtype=float,
                             elements=st.floats(-10, 10)))

    # gradient to be backpropped through this operation
    x = Tensor(x)
    y = Tensor(y)
    out = logaddexp(x, y)

    grad = data.draw(hnp.arrays(shape=out.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    # calculate logaddexp manually via mygrad-arithmetic
    x_o = Tensor(x)
    y_o = Tensor(y)
    out_o = mg.log(mg.exp(x_o) + mg.exp(y_o))

    if any(out.shape != i.shape for i in (x, y)):
        # broadcasting occurred, must reduce `out` to scalar
        # first multiply by `grad` to simulate non-trivial back-prop
        (grad * out).sum().backward()
        (grad * out_o).sum().backward()
    else:
        out.backward(grad)
        out_o.backward(grad)

    assert np.allclose(x.grad, x_o.grad), \
        "x: numerical derivative and mygrad derivative do not match"
    assert np.allclose(y.grad, y_o.grad), \
        "y: numerical derivative and mygrad derivative do not match"
