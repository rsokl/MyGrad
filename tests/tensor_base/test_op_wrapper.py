import numpy as np
import pytest

from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor


class Dummy(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        # stores output array to check memory consistency
        self.array_out = a.data * b.data
        return self.array_out


def dummy(a, b, constant=False):
    return Tensor._op(Dummy, a, b, constant=constant)


@pytest.mark.parametrize("x_const", [True, False])
@pytest.mark.parametrize("y_const", [True, False])
@pytest.mark.parametrize("op_const", [True, False])
def test_constant_arg(x_const: bool, y_const: bool, op_const: bool):
    """ test that the `constant` arg works as intended in Tensor._op"""
    x = Tensor(1, constant=x_const)
    y = Tensor(1, constant=y_const)
    out = dummy(x, y, constant=op_const)
    assert out.constant is op_const or (x_const and y_const)
    assert x._ops == {out.creator}
    assert y._ops == {out.creator}


@pytest.mark.parametrize("x_const", [True, False])
@pytest.mark.parametrize("y_const", [True, False])
@pytest.mark.parametrize("op_const", [True, False])
def test_op_wrapper_doesnt_make_copy_when_creating_output_tensor(
    x_const: bool, y_const: bool, op_const: bool
):
    x = Tensor([1.0], constant=x_const)
    y = Tensor([2.0], constant=y_const)
    out_tensor = dummy(x, y, constant=op_const)
    assert np.shares_memory(out_tensor, out_tensor.creator.array_out)
