from .multivar_operations import MultiVarBroadcastableOp
from functools import reduce

__all__ = ["MultiplySequence"]


class MultiplySequence(MultiVarBroadcastableOp):
    """ Performs f(a, b, ..., z) = a * b * ... * z"""
    def __call__(self, *input_vars):
        self.variables = input_vars
        out = reduce(lambda x, y: x*y, (var.data for var in input_vars))
        self.broadcast_check(*input_vars, out_shape=out.shape)
        return out

    def backward_var(self, grad, index):
        grad = grad * reduce(lambda x, y: x*y, (var.data for n, var in enumerate(self.variables) if n != index))
        var = self.variables[index]
        broadcasted_grad = super(MultiplySequence, self).backward_var(grad, index)
        var.backward(broadcasted_grad)
