import numpy as np

from mygrad.operation_base import Operation
from typing import Callable

__all__ = ["Reshape", "Flatten", "Squeeze", "Ravel", "ExpandDims", "BroadcastTo", "AtLeast1D", "AtLeast2D", "AtLeast3D"]

class _PreservesOrder(Operation):
    def backward_var(self, grad, index, **kwargs):
        a, = self.variables
        return np.reshape(grad, a.shape)

class _AtLeastKD(_PreservesOrder):
    can_return_view = True
    numpy_func: Callable[[np.ndarray], np.ndarray]
    
    def __call__(self, a):
        self.variables = (a,)
        return self.numpy_func(a.data)

class AtLeast1D(_AtLeastKD):
    numpy_func = staticmethod(np.atleast_1d)

class AtLeast2D(_AtLeastKD):
    numpy_func = staticmethod(np.atleast_2d)

class AtLeast3D(_AtLeastKD):
    numpy_func = staticmethod(np.atleast_3d)

class Reshape(_PreservesOrder):
    can_return_view = True

    def __call__(self, a, newshape):
        """
        Parameters
        ----------
        a : mygrad.Tensor

        newshape : Tuple[int, ...]

        Returns
        -------
        reshaped_array : numpy.ndarray
        """
        self.variables = (a,)
        return np.reshape(a.data, newshape)


class Squeeze(Operation):
    can_return_view = True

    def __call__(self, a, axis):
        """Parameters
        ----------
        axis : Optional[int, Tuple[int, ...]]"""
        self.variables = (a,)
        return np.squeeze(a.data, axis=axis)


class Flatten(Operation):
    def __call__(self, a):
        """Parameters
        ----------
        a : mygrad.Tensor"""
        self.variables = (a,)
        return a.data.flatten(order="C")


class Ravel(Operation):
    can_return_view = True

    def __call__(self, a):
        """Parameters
        ----------
        a : mygrad.Tensor"""
        self.variables = (a,)
        return np.ravel(a.data, order="C")


class ExpandDims(Operation):
    can_return_view = True

    def __call__(self, a, axis):
        """Parameters
        ----------
        a : mygrad.Tensor
        axis : int"""
        self.variables = (a,)
        return np.expand_dims(a.data, axis=axis)


class BroadcastTo(Operation):
    can_return_view = True

    def __call__(self, a, shape):
        """Parameters
        ----------
        a : mygrad.Tensor
        shape : Tuple[int, ...]"""
        self.variables = (a,)
        return np.broadcast_to(a.data, shape=shape)

    def backward_var(self, grad, index, **kwargs):
        if index != 0:  # pragma: no cover
            raise IndexError(
                f"`broadcast_to` is a unary operation. "
                f"`backward_var` was called for index {index}"
            )
        return grad
