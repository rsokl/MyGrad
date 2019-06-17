import numpy as np

from mygrad.tensor_base import Tensor

__all__ = ["argmin", "argmax"]


def argmax(a, axis=None, out=None):
    """ Returns the indices of the maximum values along an axis.

        Parameters
        ----------
        a: array_like
        
        axis: int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.
        
        out: numpy.array, optional
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
        
        Returns
        -------
        numpy.ndarray[int]"""

    a = a.data if isinstance(a, Tensor) else a
    return np.argmax(a, axis, out)


def argmin(a, axis=None, out=None):
    """ Returns the indices of the minimum values along an axis.

        Parameters
        ----------
        a: array_like
        
        axis: int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.
        
        out: numpy.array, optional
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
        
        Returns
        -------
        numpy.ndarray[int]"""

    a = a.data if isinstance(a, Tensor) else a
    return np.argmin(a, axis, out)
