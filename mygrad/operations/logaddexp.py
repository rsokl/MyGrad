from .operation_base import BroadcastableOp
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
        self.a = a
        self.b = b
        out = np.logaddexp(a.data, b.data)

        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        dLda = grad / (1 + np.exp(self.b.data - self.a.data))
        broadcasted_grad = super(Logaddexp, self).backward_a(dLda)
        self.a.backward(broadcasted_grad)

    def backward_b(self, grad):
        dLdb = grad / (1 + np.exp(self.a.data - self.b.data))
        broadcasted_grad = super(Logaddexp, self).backward_b(dLdb)
        self.b.backward(broadcasted_grad)
