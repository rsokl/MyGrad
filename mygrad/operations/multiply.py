from mygrad.operations.multivar_operations import MultiVarBroadcastableOp


class Multiply(MultiVarBroadcastableOp):
    def __call__(self, a, b):
        """ Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor"""
        self.variables = (a, b)
        out = a.data * b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            a.backward(grad * b.data, **kwargs)
        elif index == 1:  # backprop through b
            b.backward(grad * a.data, **kwargs)
