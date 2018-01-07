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
        out = np.log(np.exp(a.data) + np.exp(b.data))

        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        dLda = grad * self.a.data / (np.exp(self.a.data) + np.exp(self.b.data))
        broadcasted_grad = super(Logaddexp, self).backward_a(dLda)
        self.a.backward(broadcasted_grad)

    def backward_b(self, grad):
        dLdb = grad * self.b.data / (np.exp(self.a.data) + np.exp(self.b.data))
        broadcasted_grad = super(Logaddexp, self).backward_b(dLdb)
        self.b.backward(broadcasted_grad)
