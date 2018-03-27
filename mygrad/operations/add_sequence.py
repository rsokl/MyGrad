from .multivar_operations import BroadcastableOp

__all__ = ["AddSequence"]


class AddSequence(BroadcastableOp):
    """ Performs f(a, b, ..., z) = a + b + ... + z"""
    def __call__(self, *input_vars):
        assert len(input_vars) > 1, "`add_sequence` requires at least two operands"
        self.variables = input_vars
        out = sum(var.data for var in input_vars)
        return out

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad, **kwargs)

