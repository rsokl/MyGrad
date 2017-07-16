from .operation_base import BroadcastableOp
from .multivar_operations import MultiVarBroadcastableOp

__all__ = ["Add", "AddSequence"]


class Add(BroadcastableOp):
    def __call__(self, a, b):
        """ Performs 'add' forward-pass: f(a,b) -> a + b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """
        self.a = a
        self.b = b
        out = a.data + b.data

        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        broadcasted_grad = super(Add, self).backward_a(grad)
        self.a.backward(broadcasted_grad)

    def backward_b(self, grad):
        broadcasted_grad = super(Add, self).backward_b(grad)
        self.b.backward(broadcasted_grad)


class AddSequence(MultiVarBroadcastableOp):
    """ Performs f(a, b, ..., z) = a + b + ... + z"""
    def __call__(self, *input_vars):
        out = sum(var.data for var in input_vars)
        self.broadcast_check(*input_vars, out_shape=out.shape)
        return out

    def backward_var(self, grad, index):
        var = self.variables[index]
        broadcasted_grad = super(AddSequence, self).backward_var(grad, index)
        var.backward(broadcasted_grad)

