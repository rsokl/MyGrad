import numpy as np

from mygrad.operation_base import BroadcastableOp, Operation

__all__ = ["Reshape", "Flatten", "Squeeze", "Ravel", "ExpandDims", "BroadcastTo"]


class Reshape(Operation):
    def __call__(self, a, newshape):
        """ Parameters
            ----------
            a : mygrad.Tensor
            newshape : Tuple[int, ...]"""
        self.variables = (a,)
        return np.reshape(a.data, newshape)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return np.reshape(grad, a.shape)


class Squeeze(Operation):
    def __call__(self, a, axis):
        """ Parameters
            ----------
            axis : Optional[int, Tuple[int, ...]] """
        self.variables = (a,)
        return np.squeeze(a.data, axis=axis)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad.reshape(a.shape)


class Flatten(Operation):
    def __call__(self, a):
        """ Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return a.data.flatten(order="C")

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad.reshape(a.shape)


class Ravel(Operation):
    def __call__(self, a):
        """ Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return np.ravel(a.data, order="C")

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad.reshape(a.shape)


class ExpandDims(Operation):
    def __call__(self, a, axis):
        """ Parameters
            ----------
            a : mygrad.Tensor
            axis : int """
        self.variables = (a,)
        return np.expand_dims(a.data, axis=axis)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad.reshape(a.shape)


class BroadcastTo(BroadcastableOp):
    def __call__(self, a, shape):
        """ Parameters
            ----------
            a : mygrad.Tensor
            shape : Tuple[int, ...]"""
        self.variables = (a,)
        return np.broadcast_to(a.data, shape=shape)

    def backward_var(self, grad, index, **kwargs):
        if index != 0:  # pragma: no cover
            raise IndexError(
                "`broadcast_to` is a unary operation. "
                "`backward_var` was called for index {}".format(index)
            )
        return grad
