from ..operations.operation_base import Operation
from ..tensor_base import Tensor
import numpy as np

__all__ = ["dense"]


class Dense(Operation):
    scalar_only = True

    def __call__(self, a, b):
        assert a.ndim == 2 and b.ndim == 2
        self.a = a
        self.b = b
        return np.dot(a.data, b.data)

    def backward_a(self, grad):
        self.a.backward(np.dot(grad, self.b.data.T))

    def backward_b(self, grad):
        self.b.backward(np.dot(self.a.data.T, grad))


def dense(x, w):
    """ Perform a dense-layer pass (i.e. matrix multiplication) of a (N, D)-shape
        tensor with a (D, M)-shape tensor.

        Parameters
        ----------
        x : Union[mygrad.Tensor, array_like], shape=(N, D)

        w : Union[mygrad.Tensor, array_like], shape=(D, M)

        Returns
        -------
        Tensor, shape=(N, M)
            The result of the matrix multiplication of `x` with `w`

        Notes
        -----
        This is a "scalar-only" operation, meaning that back propagation through
        this layer assumes that a scalar (i.e. a 0-dimensional tensor) will invoke
        `tensor.backward()` for the computational graph. This is standard for a
        neural network, which terminates in a scalar loss."""
    return Tensor._op(Dense, x, w)
