import numpy as np

from mygrad.operation_base import BroadcastableOp, Operation

__all__ = ["Abs", "Sqrt", "Cbrt", "Maximum", "Minimum"]


class Abs(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.abs(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad * np.piecewise(
            a.data, [a.data < 0, a.data == 0, a.data > 0], [-1, np.nan, 1]
        )


class Sqrt(Operation):
    def __call__(self, a):
        """ f(a) = sqrt(a)

            Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return np.sqrt(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad / (2 * np.sqrt(a.data))


class Cbrt(Operation):
    def __call__(self, a):
        self.variables = (a,)
        return np.cbrt(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad / (3 * np.cbrt(a.data ** 2))


class Maximum(BroadcastableOp):
    def __call__(self, a, b):
        self.variables = (a, b)
        self.greater_than_mask = a.data > b.data
        self.equal_mask = a.data == b.data
        return np.where(self.greater_than_mask, a.data, b.data)

    def backward_var(self, grad, index, **kwargs):
        if index == 0:
            mask = self.greater_than_mask
        elif index == 1:
            mask = np.logical_not(self.greater_than_mask)
            if mask.ndim:
                np.logical_not(mask, out=mask, where=self.equal_mask)
            elif self.equal_mask:
                mask = np.logical_not(mask)
        else:
            raise IndexError

        return mask * grad


class Minimum(BroadcastableOp):
    def __call__(self, a, b):
        self.variables = (a, b)
        self.less_than_mask = a.data < b.data
        self.equal_mask = a.data == b.data
        return np.where(self.less_than_mask, a.data, b.data)

    def backward_var(self, grad, index, **kwargs):
        if index == 0:
            mask = self.less_than_mask
        elif index == 1:
            mask = np.logical_not(self.less_than_mask)
            if mask.ndim:
                np.logical_not(mask, out=mask, where=self.equal_mask)
            elif self.equal_mask:
                mask = np.logical_not(mask)
        else:
            raise IndexError

        return mask * grad
