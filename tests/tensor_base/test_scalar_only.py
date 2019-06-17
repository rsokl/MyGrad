import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from pytest import raises

from mygrad.operation_base import BroadcastableOp, Operation
from mygrad.tensor_base import Tensor


class ScalarOnlyOp(BroadcastableOp):
    def __init__(self):
        self.scalar_only = True

    def __call__(self, a, b):
        self.variables = (a, b)
        return np.array([0])


class NotScalarOnlyOp(Operation):
    def __init__(self):
        self.scalar_only = False

    def __call__(self, a, b):
        self.variables = (a, b)
        return np.array([0])


@given(
    a_const=st.booleans(),
    a_scalar_only=st.booleans(),
    b_const=st.booleans(),
    b_scalar_only=st.booleans(),
)
def test_scalar_only_op(a_const, a_scalar_only, b_const, b_scalar_only):
    """ op produces scalar_only result unless result is scalar. """
    a = Tensor(0, constant=a_const, _scalar_only=a_scalar_only)
    b = Tensor(0, constant=b_const, _scalar_only=b_scalar_only)

    out = Tensor._op(ScalarOnlyOp, a, b)
    scalar_only = True and not out.constant

    assert scalar_only is out.scalar_only

    # check out.backward()
    if scalar_only:
        with raises(Exception):
            out.backward()
    else:
        out.backward()  # a, b, out are const (nothing computed)


@given(
    a_const=st.booleans(),
    a_scalar_only=st.booleans(),
    b_const=st.booleans(),
    b_scalar_only=st.booleans(),
)
def test_standard_op(a_const, a_scalar_only, b_const, b_scalar_only):
    """ op produces standard result unless an `a` or `b` is a scalar_only variable. """
    a = Tensor(0, constant=a_const, _scalar_only=a_scalar_only)
    b = Tensor(0, constant=b_const, _scalar_only=b_scalar_only)

    scalar_only = (a.scalar_only and not a.constant) or (
        b.scalar_only and not b.constant
    )
    out = Tensor._op(NotScalarOnlyOp, a, b)

    assert scalar_only is (out.scalar_only and not out.constant)

    # check out.backward()
    if scalar_only:
        with raises(Exception):
            out.backward()
    else:
        if a.constant and b.constant:
            out.backward()  # a, b, out are const (nothing computed)
        else:
            with raises(NotImplementedError):
                out.backward()


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "truediv", "pow"])
def test_practical_scalar_only(constant, operation):
    a = Tensor([1, 2, 3], constant=constant)
    b = Tensor(3, constant=constant)
    out = getattr(a, "__" + operation + "__")(b)

    if constant:
        out.backward()
    else:
        with raises(Exception):
            out.backward()
