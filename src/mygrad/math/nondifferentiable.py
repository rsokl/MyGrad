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

def any(a, axis=None, out=None, keepdims=False):
    """ Test whether any array or Tensor element along a given axis evaluates to True.

    Returns single boolean if `axis` is ``None``

    This documentation was adapted from ``numpy.add``

    Parameters
    ----------
    a : array_like
        
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.

    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
        See `ufuncs-output-type` for more details.
    
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    
    Returns
    -------
    any : bool or ndarray
        A new boolean or `ndarray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    Tensor.any : equivalent method

    """

    a = a.data if isinstance(a, Tensor) else a
    return np.any(a, axis, out, keepdims)