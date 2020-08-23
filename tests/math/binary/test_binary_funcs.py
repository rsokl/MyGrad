""" Test all binary arithmetic operations, checks for appropriate broadcast behavior"""

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose

from mygrad import (
    add,
    arctan2,
    divide,
    logaddexp,
    logaddexp2,
    multiply,
    power,
    subtract,
)
from tests.custom_strategies import tensors

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


@given(
    t=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        elements=st.floats(-1e6, 1e6),
        constant=False,
    )
)
def test_x_pow_0_special_case(t):
    y = t ** 0
    y.backward()
    assert_allclose(y.data, np.ones_like(t))
    assert_allclose(t.grad, np.zeros_like(t))


@given(
    t=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        elements=st.floats(1e-10, 1e6),
        constant=False,
    )
)
def test_0_pow_y_special_case(t):
    y = 0 ** t
    y.backward()
    assert_allclose(y.data, np.zeros_like(t))
    assert_allclose(t.grad, np.zeros_like(t))


@fwdprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp, num_arrays=2)
def test_logaddexp_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=logaddexp,
    true_func=np.logaddexp,
    num_arrays=2,
    index_to_bnds=(-10, 10),
    use_finite_difference=True,
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
    use_finite_difference=True,
    h=1e-8,
    index_to_bnds=(-100, 100),
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
    use_finite_difference=True,
    h=1e-8,
)
def test_arctan2_bkwd():
    pass
