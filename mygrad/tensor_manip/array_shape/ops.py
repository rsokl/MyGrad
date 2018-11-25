from mygrad.operation_base import Operation
import numpy as np

__all__ = ["Reshape", "Squeeze", "Ravel"]


class Reshape(Operation):
    def __call__(self, a, shape):
        """ Parameters
            ----------
            a : mygrad.Tensor
            shape : Tuple[int, ...]"""
        self.variables = (a,)
        return a.data.reshape(shape)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad.reshape(*a.shape)


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


class Ravel(Operation):
    def __call__(self, a):
        """ Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return np.ravel(a.data, order='C')

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return grad.reshape(a.shape)
