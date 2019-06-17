""" Test all binary arithmetic operations, checks for appropriate broadcast behavior"""
import numpy as np
from hypothesis import settings
from numpy.testing import assert_allclose

from mygrad import (
    add,
    arctan2,
    divide,
    logaddexp,
    logaddexp2,
    maximum,
    minimum,
    multiply,
    power,
    subtract,
)
from mygrad.tensor_base import Tensor

from ...wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(mygrad_func=add, true_func=np.add, num_arrays=2)
def test_add_fwd():
    pass


@backprop_test_factory(mygrad_func=add, true_func=np.add, num_arrays=2)
def test_add_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=subtract, true_func=np.subtract, num_arrays=2)
def test_subtract_fwd():
    pass


@backprop_test_factory(mygrad_func=subtract, true_func=np.subtract, num_arrays=2)
def test_subtract_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=multiply, true_func=np.multiply, num_arrays=2)
def test_multiply_fwd():
    pass


@backprop_test_factory(
    mygrad_func=multiply, true_func=np.multiply, atol=1e-4, rtol=1e-4, num_arrays=2
)
def test_multiply_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=divide, true_func=np.divide, index_to_bnds={1: (1, 10)}, num_arrays=2
)
def test_divide_fwd():
    pass


@backprop_test_factory(
    mygrad_func=divide, true_func=np.divide, index_to_bnds={1: (1, 10)}, num_arrays=2
)
def test_divide_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=power,
    true_func=np.power,
    index_to_bnds={0: (1, 10), 1: (-3, 3)},
    num_arrays=2,
)
def test_power_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=power,
    true_func=np.power,
    index_to_bnds={0: (1, 10), 1: (-3, 3)},
    num_arrays=2,
)
def test_power_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp, num_arrays=2)
def test_logaddexp_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=logaddexp,
    true_func=np.logaddexp,
    num_arrays=2,
    index_to_bnds={0: (-2, 2), 1: (-2, 2)},
    finite_difference=True,
    h=1e-8,
    atol=1e-4,
    rtol=1e-4,
)
def test_logaddexp_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=logaddexp2, true_func=np.logaddexp2, num_arrays=2)
def test_logaddexp2_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=logaddexp2,
    true_func=np.logaddexp2,
    num_arrays=2,
    atol=1e-4,
    rtol=1e-4,
    finite_difference=True,
    h=1e-8,
)
def test_logaddexp2_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=arctan2, true_func=np.arctan2, num_arrays=2, index_to_bnds={1: (1, 10)}
)
def test_arctan2_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=arctan2,
    true_func=np.arctan2,
    num_arrays=2,
    atol=1e-4,
    rtol=1e-4,
    index_to_bnds={1: (1, 10)},
    finite_difference=True,
    h=1e-8,
)
def test_arctan2_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=maximum, true_func=np.maximum, num_arrays=2)
def test_maximum_fwd():
    pass


def is_not_close(arr0: Tensor, arr1: Tensor) -> bool:
    return not np.any(np.isclose(arr0.data, arr1.data))


@backprop_test_factory(
    mygrad_func=maximum, true_func=np.maximum, num_arrays=2, assumptions=is_not_close
)
def test_maximum_bkwd():
    pass


def test_maximum_bkwd_equal():
    """ regression test for documented behavior of maximum/minimum where
        x == y"""

    x = Tensor([1.0, 0.0, 2.0])
    y = Tensor([2.0, 0.0, 1.0])

    o = maximum(x, y)
    o.backward()

    assert_allclose(x.grad, [0.0, 0.0, 1])
    assert_allclose(y.grad, [1.0, 0.0, 0])
    o.null_gradients()


@fwdprop_test_factory(mygrad_func=minimum, true_func=np.minimum, num_arrays=2)
def test_minimum_fwd():
    pass


@backprop_test_factory(
    mygrad_func=minimum, true_func=np.minimum, num_arrays=2, assumptions=is_not_close
)
def test_minimum_bkwd():
    pass


def test_minimum_bkwd_equal():
    """ regression test for documented behavior of minimum/minimum where
        x == y"""

    x = Tensor([1.0, 0.0, 2.0])
    y = Tensor([2.0, 0.0, 1.0])

    o = minimum(x, y)
    o.backward()

    assert_allclose(x.grad, [1.0, 0.0, 0.0])
    assert_allclose(y.grad, [0.0, 0.0, 1.0])
    o.null_gradients()
