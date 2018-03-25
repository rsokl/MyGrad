from mygrad.operations.multivar_operations import MultiVarBroadcastableOp

__all__ = ["Subtract"]


class Subtract(MultiVarBroadcastableOp):
    def __call__(self, a, b):
        """ f(a,b) -> a - b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """
        self.variables = (a, b)
        out = a.data - b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            a.backward(grad, **kwargs)
        else:
            b.backward(grad, **kwargs)
