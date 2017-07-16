from .operation_base import BroadcastableOp
from .multivar_operations import MultiVarBroadcastableOp
from functools import reduce


class Multiply(BroadcastableOp):
    def __call__(self, a, b):
        self.a = a
        self.b = b

        out = a.data * b.data
        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        dLda = grad * self.b.data  # dL/df * df/da -> dL/da
        broadcasted_grad = super(Multiply, self).backward_a(dLda)
        self.a.backward(broadcasted_grad)

    def backward_b(self, grad):
        dLdb = grad * self.a.data  # dL/df * df/da -> dL/da
        broadcasted_grad = super(Multiply, self).backward_b(dLdb)
        self.b.backward(broadcasted_grad)


class MultiplySequence(MultiVarBroadcastableOp):
    """ Performs f(a, b, ..., z) = a + b + ... + z"""
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
