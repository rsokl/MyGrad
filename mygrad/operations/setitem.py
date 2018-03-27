from mygrad.operations.multivar_operations import BroadcastableOp
import numpy as np


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

