from mygrad.operation_base import Operation, BroadcastableOp
import numpy as np


__all__ = ["GetItem",
           "SetItem"]


class GetItem(Operation):
    def __call__(self, a, index):
        self.variables = (a,)
        self.index = index
        out = a.data[index]
        self.shape = out.shape if isinstance(out, np.ndarray) else None
        return out

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        out = np.zeros_like(a.data)
        out[self.index] = grad
        a.backward(out, **kwargs)


class SetItem(BroadcastableOp):

    def __call__(self, a, b, index):
        """ a[index] = b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor
            index : valid-array-index"""
        self.variables = (a, b)

        self.index = index
        a.data[index] = b.data

        return a.data

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            grad = np.copy(grad)
            grad[self.index] = 0
            a.backward(grad, _broadcastable=False)
        else:
            grad = grad[self.index]

            # handle the edge case of "projecting down" on setitem. E.g:
            # x = Tensor([0, 1, 2])
            # y = Tensor([3])
            # x[0] = y  # this is legal since x[0] and y have the same size
            if grad.ndim < b.ndim:
                grad = grad.reshape(b.shape)
            b.backward(grad, **kwargs)
