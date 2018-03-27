from mygrad.operations.multivar_operations import BroadcastableOp
import numpy as np


class Logaddexp(BroadcastableOp):
    def __call__(self, a, b):
        """ Performs 'logaddexp' forward-pass: f(a,b) -> log(exp(a) + exp(b))

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """
        self.variables = (a, b)
        out = np.logaddexp(a.data, b.data)
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            dLda = grad / (1 + np.exp(b.data - a.data))
            a.backward(dLda, **kwargs)
        else:
            dLdb = grad / (1 + np.exp(a.data - b.data))
            b.backward(dLdb, **kwargs)

