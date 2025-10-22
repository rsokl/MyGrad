"""Test all binary arithmetic operations, checks for appropriate broadcast behavior"""

from functools import partial
from numbers import Number
from typing import Any, Dict

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from numpy.testing import assert_allclose

from tests.custom_strategies import tensors
from tests.utils.wrappers import adds_constant_arg

from ...wrappers.uber import backprop_test_factory, fwdprop_test_factory


def inplace_op(inplace_target, other, constant=False, *, op_name: str):
    op_name = "__" + op_name + "__"

    # hack to make broadcastable shapes work for inplace op:
    _ = inplace_target.copy()

    check = False

    if np.broadcast(inplace_target, other).shape != inplace_target.shape:
        inplace_target, other = other, inplace_target
        check = True

    if check and np.broadcast(inplace_target, other).shape != inplace_target.shape:
        assume(False)

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
        **kwargs,
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


@given(
    t=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        elements=st.floats(-1e6, 1e6),
        constant=False,
    )
)
def test_x_pow_0_special_case(t):
    y = t**0
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
    y = 0**t
    y.backward()
    assert_allclose(y.data, np.zeros_like(t))
    assert_allclose(t.grad, np.zeros_like(t))
