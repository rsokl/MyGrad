import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad import Tensor
from mygrad.errors import InvalidGradient
from mygrad.operation_base import BinaryUfunc, Operation, UnaryUfunc
from tests.utils.errors import does_not_raise


class OldOperation(Operation):
    """Implements old version of MyGrad back-propagation"""

    def __call__(self, a):
        self.variables = (a,)
        return a.data

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad)


def old_op(a):
    return Tensor._op(OldOperation, a)


@given(
    constant=st.booleans(),
    arr=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=0),
        elements=st.floats(-1e6, 1e6),
    )
    | st.floats(-1e6, 1e6),
    op_before=st.booleans(),
    op_after=st.booleans(),
)
def test_backpropping_non_numeric_gradient_raises(
    constant: bool, arr: np.ndarray, op_before: bool, op_after: bool
):
    x = Tensor(arr, constant=constant)

    if op_before:
        x += 1

    x = old_op(x)

    if op_after:
        x = x * 2

    # if constant tensor, backprop should not be triggered - no exception raised
    with (pytest.raises(InvalidGradient) if not constant else does_not_raise()):
        x.backward()


def test_simple_unary_ufunc_with_where():
    from mygrad.math.exp_log.ops import Exp

    exp = Exp()
    mask = np.array([True, False, True])
    out = exp(mg.zeros((3,)), where=mask)
    assert_allclose(out[mask], [1.0, 1.0])
    assert not np.isclose(out[1].item(), 1.0)


def test_simple_binary_ufunc_with_where():
    from mygrad.math.arithmetic.ops import Multiply

    mul = Multiply()
    mask = np.array([True, False, True])
    out = mul(mg.ones((3,)), mg.full((3,), 2.0), where=mask)
    assert_allclose(out[mask], [2.0, 2.0])
    assert not np.isclose(out[1].item(), 2.0)


def test_simple_sequential_func_with_where():
    from mygrad.math.sequential.ops import Sum

    sum_ = Sum()
    assert sum_(mg.ones((3,)), where=np.array([True, False, True])).item() == 2.0
