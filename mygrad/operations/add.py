from .multivar_operations import MultiVarBroadcastableOp

__all__ = ["Add"]


class Add(MultiVarBroadcastableOp):
    def __call__(self, a, b):
        """ Performs 'add' forward-pass: f(a,b) -> a + b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """

        self.variables = (a, b)
        out = a.data + b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad, **kwargs)


