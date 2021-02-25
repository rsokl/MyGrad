""" Test all binary arithmetic operations, checks for appropriate broadcast behavior"""
from functools import partial
from numbers import Number
from typing import Any, Dict

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
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
from tests.utils.wrappers import adds_constant_arg

from ...wrappers.uber import backprop_test_factory, fwdprop_test_factory


def inplace_op(inplace_target, other, constant=False, *, op_name: str):

    op_name = "__" + op_name + "__"

    # hack to make broadcastable shapes work for inplace op:
    # 1. Ensure that inplace op has at least as many items as `other`.
    # 2. `other` can't have excess leading dims
    inplace_target, other = (
        (inplace_target, other)
        if inplace_target.size >= other.size
        else (other, inplace_target)
    )
    if other.ndim > inplace_target.ndim:
        other = other[(0,) * (other.ndim - inplace_target.ndim)]

    # touch so that it doesn't look like the input
    # was mutated
    inplace_target = +inplace_target
    if isinstance(inplace_target, Number):
        inplace_target = np.asarray(inplace_target)

    adds_constant_arg(getattr(inplace_target, op_name))(other, constant=constant)

    return inplace_target


ipow = partial(inplace_op, op_name="ipow")
idiv = partial(inplace_op, op_name="itruediv")


@pytest.mark.parametrize(
    ("op_name", "kwargs"),
    [
        ("iadd", dict()),
        ("isub", dict()),
        ("imul", dict()),
        ("itruediv", dict(index_to_bnds={0: (1, 100), 1: (1, 100)})),
        ("ipow", dict(index_to_bnds={0: (1, 4), 1: (1, 4)})),
    ],
)
def test_inplace_arithmetic_fwd(op_name: str, kwargs: Dict[str, Any]):
    iop = partial(inplace_op, op_name=op_name)

    @fwdprop_test_factory(
        mygrad_func=iop,
        true_func=iop,
        num_arrays=2,
        permit_0d_array_as_float=False,
        **kwargs
    )
    def iop_fwd():
        pass

    iop_fwd()


@pytest.mark.parametrize(
    ("op_name", "kwargs"),
    [
        ("iadd", dict()),
        ("isub", dict()),
        ("imul", dict()),
        ("itruediv", dict(index_to_bnds={0: (1, 100), 1: (1, 100)})),
        ("ipow", dict(index_to_bnds={0: (1, 4), 1: (1, 4)})),
    ],
)
def test_inplace_arithmetic_bkwd(op_name: str, kwargs: Dict[str, Any]):
    iop = partial(inplace_op, op_name=op_name)

    @backprop_test_factory(
        mygrad_func=iop, true_func=iop, num_arrays=2, vary_each_element=True, **kwargs
    )
    def iop_bkwd():
        pass

    iop_bkwd()


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
