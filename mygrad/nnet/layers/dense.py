from mygrad.operations.multivar_operations import MultiVarOperation
from ...tensor_base import Tensor
import numpy as np


class Dense(MultiVarOperation):
    scalar_only = True

    def __call__(self, a, b):
        self.variables = (a, b)
        assert b.ndim == 2
        assert 3 >= a.ndim >= 2

        return np.dot(a.data, b.data)

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            if a.ndim == 2:
                a.backward(np.dot(grad, b.data.T))
            else:
                # grad: (T, N, D)
                # b: (C, D)
                a.backward(np.einsum("ijk, nk", grad, b.data), **kwargs)
        else:
            if a.ndim == 2:
                b.backward(np.dot(a.data.T, grad))
            else:
                # a: (T, N, C)
                # grad: (T, N, D)
                b.backward(np.tensordot(a.data, grad, ((0, 1), (0, 1))), **kwargs)


def dense(x, w):
    """ Perform a dense-layer pass (i.e. matrix multiplication) of a (N, D)-shape
        tensor with a (D, M)-shape tensor.

        Parameters
        ----------
        x : Union[mygrad.Tensor, array_like], shape=(N, D) or (T, N, D)

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
