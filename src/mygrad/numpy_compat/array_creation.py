import numpy as np

from mygrad.tensor_base import Tensor

__all__ = ["asarray"]


def asarray(a, dtype=None, order=None) -> np.ndarray:
    """Convert the input to an array.

    This docstring is adapted from that of ``numpy.asarray``

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.

    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.

    ascontiguousarray : Convert input to a contiguous array.

    asfarray : Convert input to a floating point ndarray.

    asfortranarray : Convert input to an ndarray with column-major
                     memory order.

    asarray_chkfinite : Similar function which checks input for NaNs and Infs.

    fromiter : Create an array from an iterator.

    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Convert a tensor into an array. No copy of the
    underlying numpy array is created:

    >>> t = mg.Tensor([1, 2.])
    >>> np.asarray(t)
    array([1., 2.])
    >>> t.data is np.asarray(t))
    True

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.asarray(a, dtype=np.float32) is a
    True
    >>> np.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True
    """
    if isinstance(a, Tensor):
        a = a.data
    return np.asarray(a, dtype=dtype, order=order)
