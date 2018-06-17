from mygrad.operation_base import Operation
import numpy as np

__all__ = ["Reshape", "Squeeze"]


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
        a.backward(grad.reshape(*a.shape), **kwargs)


class Squeeze(Operation):
    def __call__(self, a, axis):
        """ Parameters
            ----------
            axis : Optional[int, Tuple[int, ...]] """
        self.variables = (a,)
        return np.squeeze(a.data, axis=axis)
    
    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad.reshape(a.shape), **kwargs)