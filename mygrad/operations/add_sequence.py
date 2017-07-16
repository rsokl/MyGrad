from .multivar_operations import MultiVarBroadcastableOp

__all__ = ["AddSequence"]


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