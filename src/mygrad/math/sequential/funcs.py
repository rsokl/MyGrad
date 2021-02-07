from mygrad.tensor_base import Tensor

from .ops import *

__all__ = [
    "sum",
    "mean",
    "var",
    "std",
    "amax",
    "amin",
    "max",
    "min",
    "prod",
    "cumprod",
    "cumsum",
]


def sum(x, axis=None, keepdims=False, constant=False):
    """
    Sum of tensor elements over a given axis.

    Parameters
    ----------
    x : array_like

    axis : Optional[int, Tuple[ints, ...]]
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input tensor.  If
        axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input tensor.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    sum_along_axis : mygrad.Tensor
        A Tensor with the same shape as `self`, with the specified
        axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
        a 0-dim Tensor is returned.

    See Also
    --------
    mygrad.Tensor.sum : Equivalent method.

    cumsum : Cumulative sum of array elements.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    The sum of an empty tensor is the neutral element 0:

    >>> mygrad.sum([])
    Tensor(0.0)

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> mg.sum([0.5, 1.5])
    Tensor(2.0)
    >>> mg.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    Tensor(1)
    >>> mg.sum([[0, 1], [0, 5]])
    Tensor(6)
    >>> mg.sum([[0, 1], [0, 5]], axis=0)
    Tensor([0, 6])
    >>> mg.sum([[0, 1], [0, 5]], axis=1)
    Tensor([1, 5])

    If the accumulator is too small, overflow occurs:

    >>> mg.ones(128, dtype=mg.int8).sum(dtype=np.int8)
    Tensor(-128)

    You can also start the sum with a value other than zero:

    >>> mg.sum([10], initial=5)
    Tensor(15)
    """
    return Tensor._op(
        Sum, x, op_kwargs=dict(axis=axis, keepdims=keepdims), constant=constant
    )


def mean(x, axis=None, keepdims=False, constant=False):
    """
    Mean of tensor elements over a given axis.

    Parameters
    ----------
    x : array_like

    axis : Optional[int, Tuple[ints, ...]
        Axis or axes along which a mean is performed.  The default,
        axis=None, will mean all of the elements of the input tensor.  If
        axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a mean is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input tensor.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mean_along_axis : Tensor
        A Tensor with the same shape as `self`, with the specified
        axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
        a 0-dim Tensor is returned.

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> a = mg.Tensor([[1, 2],
    ...                [3, 4]])
    >>> mg.mean(a)
    Tensor(2.5)
    >>> mg.mean(a, axis=0)
    Tensor([ 2.,  3.])
    >>> mg.mean(a, axis=1)
    Tensor([ 1.5,  3.5])

    In single precision, `mean` can be inaccurate:

    >>> a = mg.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> mg.mean(a)
    Tensor(0.54999924)

    Computing the mean in float64 is more accurate:

    >>> mg.mean(a, dtype=np.float64)
    Tensor(0.55000000074505806)
    """
    return Tensor._op(
        Mean, x, op_kwargs=dict(axis=axis, keepdims=keepdims), constant=constant
    )


def var(x, axis=None, ddof=0, keepdims=False, constant=False):
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose variance is desired.

    axis : Optional[int, Tuple[int, ...]]
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.

    ddof : int, optional (default=0)
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.

    keepdims : bool, optional (default=False)
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array..

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    variance : mygrad.Tensor

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> a = mg.Tensor([[1, 2],
    ...                [3, 4]])
    >>> mg.var(a)
    Tensor(1.25)
    >>> mg.var(a, axis=0)
    Tensor([ 1.,  1.])
    >>> mg.var(a, axis=1)
    Tensor([ 0.25,  0.25])

    In single precision, ``var()`` can be inaccurate:

    >>> a = mg.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> mg.var(a)
    Tensor(0.20250003)

    Computing the variance in float64 is more accurate:

    >>> mg.var(a, dtype=np.float64)
    Tensor(0.20249999932944759)
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    Tensor(0.2025)
    """
    return Tensor._op(
        Variance,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, ddof=ddof),
        constant=constant,
    )


def std(x, axis=None, ddof=0, keepdims=False, constant=False):
    """
    Compute the standard deviation along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose standard deviation is desired.

    axis : Optional[int, Tuple[int, ...]]
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.

    ddof : int, optional (default=0)
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.

    keepdims : bool, optional (default=False)
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    std : mygrad.Tensor

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> a = mg.Tensor([[1, 2],
    ...                [3, 4]])
    >>> mg.std(a)
    Tensor(1.1180339887498949)
    >>> mg.std(a, axis=0)
    Tensor([ 1.,  1.])
    >>> mg.std(a, axis=1)
    Tensor([ 0.5,  0.5])

    In single precision, ``var()`` can be inaccurate:

    >>> a = mg.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> mg.std(a)
    Tensor(0.45000005)

    Computing the variance in float64 is more accurate:

    >>> mg.std(a, dtype=np.float64)
    Tensor(0.44999999925494177)
    """
    return Tensor._op(
        StdDev,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, ddof=ddof),
        constant=constant,
    )


def max(x, axis=None, keepdims=False, constant=False):
    """
    Return the maximum of a tensor or maximum along its axes.

    Parameters
    ----------
    x : array_like

    axis : Optional[int, Tuple[int, ...]]
        Axis or axes along which to operate. By default, flattened input is used.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    max : mygrad.Tensor
        Maximum of `a`. If `axis` is None, the result is a 0-D tensor.

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> a = mg.arange(4).reshape((2,2))
    >>> a
    Tensor([[0, 1],
            [2, 3]])
    >>> mg.amax(a)           # Maximum of the flattened array
    Tensor(3)
    >>> mg.amax(a, axis=0)   # Maxima along the first axis
    Tensor([2, 3])
    >>> mg.amax(a, axis=1)   # Maxima along the second axis
    Tensor([1, 3])
    >>> b = mg.arange(5, dtype=float)
    >>> b[2] = np.NaN
    >>> mg.amax(b)
    Tensor(nan)
    """
    return Tensor._op(
        MaxMin,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin="max"),
        constant=constant,
    )


def min(x, axis=None, keepdims=False, constant=False):
    """
    Return the minimum of a tensor or minimum along its axes.

    Parameters
    ----------
    axis : Optional[int, Tuple[int, ...]]
        Axis or axes along which to operate. By default, flattened input is used.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    min : mygrad.Tensor
        Minimum of `a`. If `axis` is None, the result is a 0-D tensor.

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> a = mg.arange(4).reshape((2,2))
    >>> a
    Tensor([[0, 1],
            [2, 3]])
    >>> mg.amin(a)           # Minimum of the flattened array
    Tensor(0)
    >>> mg.amin(a, axis=0)   # Minima along the first axis
    Tensor([0, 1])
    >>> mg.amin(a, axis=1)   # Minima along the second axis
    Tensor([0, 2])
    >>> b = mg.arange(5, dtype=float)
    >>> b[2] = np.NaN
    >>> mg.amin(b)
    Tensor(nan)
    """
    return Tensor._op(
        MaxMin,
        x,
        op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin="min"),
        constant=constant,
    )


# aliases
amin = min
amax = max


def prod(a, axis=None, keepdims=False, constant=False):
    """
    Return the product of array elements over given axes.

    Parameters
    ----------
    a : array_like
        Input data.

    axis : Optional[int, Tuple[int, ...]]
        Axis or axes along which to operate. By default, flattened input is used.

    keepdims : bool, optional (default=False)
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

    Returns
    -------
    product_along_axis : mygrad.Tensor
        A tensor shaped as `a` but with the specified axis removed.

    Notes
    -----
    The product of an empty tensor is the neutral element 1:

    >>> import mygrad
    >>> mygrad.prod([])
    Tensor(1.0)

    Examples
    --------
    By default, calculate the product of all elements:

    >>> import mygrad as mg
    >>> mg.prod([1.,2.])
    Tensor(2.0)

    Even when the input array is two-dimensional:

    >>> mg.prod([[1.,2.],
    ...          [3.,4.]])
    Tensor(24.0)

    But we can also specify the axis over which to multiply:

    >>> mg.prod([[1.,2.],
    ...          [3.,4.]], axis=1)
    Tensor([  2.,  12.])"""
    return Tensor._op(
        Prod, a, op_kwargs=dict(axis=axis, keepdims=keepdims), constant=constant
    )


def cumprod(a, axis=None, constant=False):
    """
    Return the cumulative product of elements along a given axis.

    This docstring was adapted from the official numpy documentation

    Parameters
    ----------
    a : array_like
        Input array.

    axis : Optional[int]
        Axis along which the cumulative product is computed.  By default
        the input is flattened.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> from mygrad import cumprod, Tensor
    >>> a = Tensor([[1, 2, 3],
    ...             [4, 5, 6]])

    >>> cumprod(a)
    Tensor([  1   2   6  24 120 720])

    The cumulative product for each column (i.e., over the rows) of `a`:

    >>> cumprod(a, axis=0)
    Tensor([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of `a`:

    >>> cumprod(a, axis=1)
    Tensor([[  1,   2,   6],
            [  4,  20, 120]])"""

    return Tensor._op(CumProd, a, op_kwargs=dict(axis=axis), constant=constant)


def cumsum(a, axis=None, constant=False):
    """
    Return the cumulative sum of the elements along a given axis.

    This docstring was adapted from the official numpy documentation

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Examples
    --------
    >>> from mygrad import cumsum, Tensor
    >>> a = Tensor([[1, 2, 3],
    ...             [4, 5, 6]])
    >>> cumsum(a)
    Tensor([ 1,  3,  6, 10, 15, 21])

    >>> cumsum(a, axis=0)      # sum over rows for each of the 3 columns
    Tensor([[1, 2, 3],
            [5, 7, 9]])
    >>> cumsum(a, axis=1)      # sum over columns for each of the 2 rows
    Tensor([[ 1,  3,  6],
            [ 4,  9, 15]])
    """

    return Tensor._op(CumSum, a, op_kwargs=dict(axis=axis), constant=constant)
