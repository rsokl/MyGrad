from .operation_base import BroadcastableOp

__all__ = ["Subtract"]


class Subtract(BroadcastableOp):
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
        out = a.data - b.data

        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        broadcasted_grad = super(Subtract, self).backward_a(grad)
        self.a.backward(broadcasted_grad)

    def backward_b(self, grad):
        broadcasted_grad = super(Subtract, self).backward_b(-1 * grad)
        self.b.backward(broadcasted_grad)
