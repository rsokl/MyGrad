import hypothesis.strategies as st
from hypothesis import given
from pytest import raises

from mygrad.tensor_base import Tensor
from mygrad.operations.operation_base import Operation

import numpy as np


class ScalarOnlyOp(Operation):
    def __init__(self):
        self.scalar_only = True

    def __call__(self, a, b=None):
        self.a = a
        if b is not None:
            self.b = b
        return np.array([0])


class NotScalarOnlyOp(Operation):
    def __init__(self):
        self.scalar_only = False

    def __call__(self, a, b=None):
        self.a = a
        if b is not None:
            self.b = b
        return np.array([0])


@given(a_const=st.booleans(), a_scalar_only=st.booleans(),
       b_const=st.booleans(), b_scalar_only=st.booleans())
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


@given(a_const=st.booleans(), a_scalar_only=st.booleans(),
       b_const=st.booleans(), b_scalar_only=st.booleans())
def test_standard_op(a_const, a_scalar_only, b_const, b_scalar_only):
    """ op produces standard result unless an `a` or `b` is a scalar_only variable. """
    a = Tensor(0, constant=a_const, _scalar_only=a_scalar_only)
    b = Tensor(0, constant=b_const, _scalar_only=b_scalar_only)

    scalar_only = (a.scalar_only and not a.constant) or (b.scalar_only and not b.constant)
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


@given(a_const=st.booleans(), a_scalar_only=st.booleans())
def test_scalar_only_monop(a_const, a_scalar_only):
    """ mon_op produces scalar_only result unless result is scalar. """
    a = Tensor(0, constant=a_const, _scalar_only=a_scalar_only)

    out = Tensor._op(ScalarOnlyOp, a)
    scalar_only = True and not a.constant

    assert scalar_only is out.scalar_only

    # check out.backward()
    if scalar_only:
        with raises(Exception):
            out.backward()
    else:
        out.backward()  # a and out are const (nothing computed)


@given(a_const=st.booleans(), a_scalar_only=st.booleans())
def test_standard_monop(a_const, a_scalar_only):
    """ mon_op produces standard result unless `a` is a scalar_only variable. """
    a = Tensor(0, constant=a_const, _scalar_only=a_scalar_only)

    out = Tensor._op(NotScalarOnlyOp, a)
    scalar_only = a.scalar_only and not a.constant

    assert scalar_only is out.scalar_only

    # check out.backward()
    if scalar_only:
        with raises(Exception):
            out.backward()
    else:
        if a.constant:
            out.backward()  # a, b, out are const (nothing computed)
        else:
            with raises(NotImplementedError):
                out.backward()


def test_practical_scalar_only():
    a = Tensor([1, 2, 3])
    b = Tensor(3)
    out = a + b
    with raises(Exception):
        out.backward()

    a = Tensor([1, 2, 3])
    b = Tensor(3)
    out = a * b
    with raises(Exception):
        out.backward()

    a = Tensor([1, 2, 3])
    b = Tensor(3)
    out = a - b
    with raises(Exception):
        out.backward()

    a = Tensor([1, 2, 3])
    b = Tensor(3)
    out = a / b
    with raises(Exception):
        out.backward()

    a = Tensor([1, 2, 3])
    b = Tensor(3)
    out = a ** b
    with raises(Exception):
        out.backward()

    a = Tensor([1, 2, 3], constant=True)
    b = Tensor(3, constant=True)
    out = a + b
    out.backward()
