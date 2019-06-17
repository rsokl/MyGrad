from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor


class Dummy(Operation):
    def __call__(self, a, b):
        self.variables = (a, b)
        return 1
    

def dummy(a, b, constant=False):
    return Tensor._op(Dummy, a, b, constant=constant)


def test_constant_arg():
    """ test that the `constant` arg works as intended in Tensor._op"""
    a = Tensor(1)
    b = Tensor(1)
    o_true = dummy(a, b, constant=True)
    assert o_true.constant is True
    assert a._ops == set()
    assert b._ops == set()

    o_false = dummy(a, b, constant=False)
    assert o_false.constant is False
    assert a._ops == {o_false.creator}
    assert b._ops == {o_false.creator}
