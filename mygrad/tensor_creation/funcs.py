import numpy as np

from mygrad.tensor_base import Tensor

__all__ = ["empty",
           "empty_like",
           "eye",
           "identity",
           "ones",
           "ones_like",
           "zeros",
           "zeros_like",
           "full",
           "full_like",
           "arange",
           "linspace",
           "logspace",
           "geomspace"]


def empty(shape, dtype=np.float32, constant=False):
    """ Return a new Tensor of the given shape and type, without initializing entries.

        Parameters
        ----------
        shape : Union[int, Tuple[int]]
            The shape of the empty array.

        dtype : data-type, optional (default=numpy.float32)
            The data type of the output Tensor.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A tensor of uninitialized data of the given shape and dtype.

        See Also
        --------
        empty_like : Return an empty tensor with shape and type of input.
        ones : Return a new tensor setting values to one.
        zeros : Return a new tensor setting values to zero.
        full : Return a new tensor of given shape filled with value.
    
    
        Notes
        -----
        `empty`, unlike `zeros`, does not set the array values to zero,
        and may therefore be marginally faster.  On the other hand, it requires
        the user to manually set all the values in the array, and should be
        used with caution.
    
        Examples
        --------
        >>> import mygrad as mg
        >>> mg.empty([2, 2], constant=True)
        Tensor([[ -9.74499359e+001,   6.69583040e-309],
                [  2.13182611e-314,   3.06959433e-309]])         #random
    
        >>> mg.empty([2, 2], dtype=int)
        Tensor([[-1073741821, -1067949133],
                [  496041986,    19249760]])                     #random
    """
    return Tensor(np.empty(shape, dtype), constant=constant)


def empty_like(other, dtype=None, constant=False):
    """ Return a new Tensor of the same shape and type as the given array.

        Parameters
        ----------
        other : Union[Tensor, ArrayLike]
            The Tensor or array whose shape and datatype should be mirrored.

        dtype : data-type, optional (default=None)
            Override the data type of the returned Tensor with this value, or None to not override.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A tensor of uninitialized data whose shape and type match `other`.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.arange(4).reshape
        >>> mg.empty(x, constant=True)
        Tensor([[ -9.74499359e+001,   6.69583040e-309],
                [  2.13182611e-314,   3.06959433e-309]])         #random

        >>> mg.empty(x, dtype=int)
        Tensor([[-1073741821, -1067949133],
                [  496041986,    19249760]])                     #random
    """
    if isinstance(other, Tensor):
        other = other.data
        
    return Tensor(np.empty_like(other, dtype), constant=constant)


def eye(rows, cols=None, diag_idx=0, dtype=np.float32, constant=False):
    """ Return a 2D Tensor with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        rows : int
            The number of rows in the output Tensor.

        cols : int, optional (default=None)
            The number of columns in the output, or None to match `rows`.

        diag_idx : int, optional (default=0)
            The index of the diagonal. 0 is the main diagonal; a positive value is the upper
            diagonal, while a negative value refers to the lower diagonal.

        dtype : data-type, optional (default=numpy.float32)
            The data type of the output Tensor.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A tensor whose elements are 0, except for the :math:`k`-th diagonal, whose values are 1.

        Examples
        --------
        >>> import mygrad as mg
        >>> mg.eye(2, dtype=int)
        Tensor([[1, 0],
                [0, 1]])
        >>> mg.eye(3, k=1)
        Tensor([[ 0.,  1.,  0.],
                [ 0.,  0.,  1.],
                [ 0.,  0.,  0.]])
    """
    return Tensor(np.eye(rows, cols, diag_idx, dtype), constant=constant)


def identity(n, dtype=np.float32, constant=False):
    """ Return the identity Tensor; a square Tensor with 1s on the main diagonal and 0s elsewhere.

        Parameters
        ----------
        n : int
            The number of rows and columns in the output Tensor.

        dtype : data-type, optional (default=numpy.float32)
            The data type of the output Tensor.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A square Tensor whose main diagonal is 1 and all other elements are 0.

        Examples
        --------
        >>> importy mygrad as mg
        >>> mg.identity(3)
        Tensor([[ 1.,  0.,  0.],
                [ 0.,  1.,  0.],
                [ 0.,  0.,  1.]])
    """
    return Tensor(np.identity(n, dtype), constant=constant)
    

def ones(shape, dtype=np.float32, constant=False):
    """ 
    Return a Tensor of the given shape and type, filled with ones.

    Parameters
    ----------
    shape : Union[int, Tuple[int]]
        The shape of the output Tensor.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

    Returns
    -------
    Tensor
        A Tensor of ones with the given shape and data type.


    See Also
    --------
    ones_like : Return an tensor of ones with shape and type of input.
    empty : Return a new uninitialized tensor.
    zeros : Return a new tensor setting values to zero.
    full : Return a new tensor of given shape filled with value.
    
    Examples
    --------
    >>> import mygrad as mg
    >>> mg.ones(5)
    Tensor([ 1.,  1.,  1.,  1.,  1.])

    >>> mg.ones((5,), dtype=int)
    Tensor([1, 1, 1, 1, 1])

    >>> mg.ones((2, 1))
    Tensor([[ 1.],
           [ 1.]])

    >>> mg.ones((2, 2))
    Tensor([[ 1.,  1.],
            [ 1.,  1.]])
    """
    return Tensor(np.ones(shape, dtype), constant=constant)


def ones_like(other, dtype=None, constant=False):
    """
    Return a Tensor of the same shape and type as the given, filled with ones.

    Parameters
    ----------
    other : array_like
        The Tensor or array whose shape and datatype should be mirrored.

    dtype : data-type, optional (default=None)
        Override the data type of the returned Tensor with this value, or None to not override.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

    Returns
    -------
    Tensor
        A Tensor of ones whose shape and data type match `other`.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.arange(6).reshape((2, 3))
    >>> x
    Tensor([[0, 1, 2],
            [3, 4, 5]])

    >>> mg.ones_like(x)
    Tensor([[1, 1, 1],
            [1, 1, 1]])

    >>> y = mg.arange(3, dtype=float)
    >>> y
    Tensor([ 0.,  1.,  2.])

    >>> mg.ones_like(y)
    Tensor([ 1.,  1.,  1.])
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.ones_like(other, dtype), constant=constant)


def zeros(shape, dtype=np.float32, constant=False):
    """ 
    Return a Tensor of the given shape and type, filled with zeros.

    Parameters
    ----------
    shape : Union[int, Tuple[int]]
        The shape of the output Tensor.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

    Returns
    -------
    Tensor
        A Tensor of zeros with the given shape and data type.

    See Also
    --------
    ones_like : Return an tensor of ones with shape and type of input.
    empty : Return a new uninitialized tensor.
    ones : Return a new tensor setting values to one.
    full : Return a new tensor of given shape filled with value.

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.zeros(5)
    Tensor([ 0.,  0.,  0.,  0.,  0.])

    >>> mg.zeros((5,), dtype=int, constant=True) # tensor will not back-propagate a gradient
    Tensor([0, 0, 0, 0, 0])

    >>> mg.zeros((2, 1))
    Tensor([[ 0.],
            [ 0.]])

    >>> mg.zeros((2, 2))
    Tensor([[ 0.,  0.],
            [ 0.,  0.]])
    """
    return Tensor(np.zeros(shape, dtype), constant=constant)


def zeros_like(other, dtype=None, constant=False):
    """ 
    Return a Tensor of the same shape and type as the given, filled with zeros.

    Parameters
    ----------
    other : Union[Tensor, ArrayLike]
        The Tensor or array whose shape and datatype should be mirrored.

    dtype : data-type, optional (default=None)
        Override the data type of the returned Tensor with this value, or None to not override.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

    Returns
    -------
    Tensor
        A Tensor of zeros whose shape and data type match `other`.

    See Also
    --------
    empty_like : Return an empty tensor with shape and type of input.
    ones_like : Return an tensor of ones with shape and type of input.
    full_like : Return a new tensor with shape of input filled with value.
    zeros : Return a new tensor setting values to zero.
    
    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.arange(6).reshape((2, 3))
    >>> x
    Tensor([[0, 1, 2],
            [3, 4, 5]])

    >>> mg.zeros_like(x, constant=True)  # tensor will not back-propagate a gradient
    Tensor([[0, 0, 0],
            [0, 0, 0]])

    >>> y = mg.arange(3, dtype=float)
    >>> y
    Tensor([ 0.,  1.,  2.])

    >>> mg.zeros_like(y)
    Tensor([ 0.,  0.,  0.])
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.zeros_like(other, dtype), constant=constant)


def full(shape, fill_value, dtype=None, constant=False):
    """ 
    Return a Tensor of the given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : Union[int, Tuple[int]]
        The shape of the output Tensor.

    fill_value : Real
        The value with which to fill the output Tensor.

    dtype : data-type, optional (default=None)
        The data type of the output Tensor, or None to match `fill_value`..

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

    Returns
    -------
    Tensor
        A Tensor of `fill_value` with the given shape and dtype.

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.full((2, 2), 33)
    Tensor([[ 33,  33],
            [ 33,  33]])

    >>> mg.full((2, 2), 10)
    Tensor([[10, 10],
            [10, 10]])
    """
    return Tensor(np.full(shape, fill_value, dtype), constant=constant)


def full_like(other, fill_value, dtype=None, constant=False):
    """ Return a Tensor of the same shape and type as the given, filled with `fill_value`.

        Parameters
        ----------
        other : Union[Tensor, ArrayLike]
            The Tensor or array whose shape and datatype should be mirrored.

        dtype : data-type, optional (default=None)
            Override the data type of the returned Tensor with this value, or None to not override.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A Tensor of `fill_value` whose shape and data type match `other`.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.arange(6, dtype=int)
        >>> mg.full_like(x, 1)
        Tensor([1, 1, 1, 1, 1, 1])
        >>> mg.full_like(x, 0.1)
        Tensor([0, 0, 0, 0, 0, 0])
        >>> mg.full_like(x, 0.1, dtype=np.double)
        Tensor([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
        >>> mg.full_like(x, np.nan, dtype=np.double)
        Tensor([ nan,  nan,  nan,  nan,  nan,  nan])
    
        >>> y = mg.arange(6, dtype=np.double)
        >>> mg.full_like(y, 0.1)
        Tensor([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.full_like(other, fill_value, dtype), constant=constant)


def arange(stop, start=0, step=1, dtype=None, constant=False):
    """ Return a Tensor with evenly-spaced values within a given interval.

        Values are generated within [start, stop). Note that for non-integer steps, results may be
        inconsistent; you are better off using `linspace` instead.

        Parameters
        ----------
        start : Real, optional, default=0
            The start of the interval, inclusive.

        stop : Real
            The end of the interval, exclusive.

        step : Real, optional (default=1)
            The spacing between successive values.

        dtype : data-type, optional (default=None)
            The data type of the output Tensor, or None to infer from the inputs.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A Tensor of evenly-spaced values in [start, end).
        
        Examples
        --------
        >>> import mygrad as mg
        >>> mg.arange(3)
        Tensor([0, 1, 2])
        >>> mg.arange(3.0, constant=True)
        Tensor([ 0.,  1.,  2.])  # resulting tensor will not back-propagate a gradient
        >>> mg.arange(3,7)
        Tensor([3, 4, 5, 6])
        >>> mg.arange(3,7,2)
        Tensor([3, 5])
    """
    if start > stop:
        tmp = start
        start = stop
        stop = tmp
    return Tensor(np.arange(start, stop, step, dtype), constant=constant)


def linspace(start, stop, num=50, include_endpoint=True, dtype=None, constant=False):
    """ Return a Tensor with evenly-spaced numbers over a specified interval.

        Values are generated within [start, stop], with the endpoint optionally excluded.

        Parameters
        ----------
        start : Real
            The starting value of the sequence, inclusive.

        stop : Real
            The ending value of the sequence, inclusive unless `include_endpoint` is False.

        num : int, optional (default=50)
            The number of values to generate. Must be non-negative.

        include_endpoint : bool, optional (default=True)
            Whether to include the endpoint in the Tensor. Note that if False, the step size changes
            to accommodate the sequence excluding the endpoint.

        dtype : data-type, optional (default=None)
            The data type of the output Tensor, or None to infer from the inputs.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        Tensor
            A Tensor of `num` evenly-spaced values in [start, stop] or [start, stop), depending on
            `include_endpoint`.

        See Also
        --------
        arange : Similar to `linspace`, but uses a step size (instead of the
                 number of samples).
        logspace : Samples uniformly distributed in log space.

        Examples
        --------
        >>> import mygrad as mg
        >>> mg.linspace(2.0, 3.0, num=5)
        Tensor([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
        >>> mg.linspace(2.0, 3.0, num=5, endpoint=False)
        Tensor([ 2. ,  2.2,  2.4,  2.6,  2.8])
    """
    return Tensor(np.linspace(start, stop, num, include_endpoint, dtype=dtype), constant=constant)


def logspace(start, stop, num=50, include_endpoint=True, base=10, dtype=None, constant=False):
    """ Return a Tensor with evenly-spaced numbers over a specified interval on a log scale.

        In linear space, values are generated within [base**start, base**stop], with the endpoint
        optionally excluded.

        Parameters
        ----------
        start : Real
            The starting value of the sequence, inclusive; start at `base ** start`.

        stop : Real
            The ending value of the sequence, inclusive unless `include_endpoint` is False; end at
            `base ** stop`.

        num : int, optional (default=50)
            The number of values to generate. Must be non-negative.

        include_endpoint : bool, optional (default=True)
            Whether to include the endpoint in the Tensor. Note that if False, the step size changes
            to accommodate the sequence excluding the endpoint.

        base : Real, optional (default=10)
            The base of the log space.

        dtype : data-type, optional (default=None)
            The data type of the output Tensor, or None to infer from the inputs.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)
    
        See Also
        --------
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples. Note that, when used with a float endpoint, the
                 endpoint may or may not be included.
        linspace : Similar to logspace, but with the samples uniformly distributed
                   in linear space, instead of log space.
        geomspace : Similar to logspace, but with endpoints specified directly.
    
        Examples
        --------
        >>> import mygrad as mg
        >>> mg.logspace(2.0, 3.0, num=4)
        Tensor([  100.        ,   215.443469  ,   464.15888336,  1000.        ])
        >>> mg.logspace(2.0, 3.0, num=4, endpoint=False)
        Tensor([ 100.        ,  177.827941  ,  316.22776602,  562.34132519])
        >>> mg.logspace(2.0, 3.0, num=4, base=2.0)
        Tensor([ 4.        ,  5.0396842 ,  6.34960421,  8.        ])
        
        Returns
        -------
        Tensor
            A Tensor of `num` evenly-spaced values in the log interval [base**start, base**stop].
    """
    return Tensor(np.logspace(start, stop, num, include_endpoint, base, dtype), constant=constant)


def geomspace(start, stop, num=50, include_endpoint=True, dtype=None, constant=False):
    """ Return a Tensor with evenly-spaced values in a geometric progression.

        Each output sample is a constant multiple of the previous output.

        Parameters
        ----------
        start : Real
            The starting value of the output.

        stop : Real
            The ending value of the sequence, inclusive unless `include_endpoint` is false.

        num : int, optional (default=50)
            The number of values to generate. Must be non-negative.

        include_endpoint : bool, optional (default=True)
            Whether to include the endpoint in the Tensor. Note that if False, the step size changes
            to accommodate the sequence excluding the endpoint.

        dtype : data-type, optional (default=None)
            The data type of the output Tensor, or None to infer from the inputs.

        constant : bool, optional (default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)
            
        Returns
        -------
        Tensor
            A Tensor of `num` samples, evenly-spaced in a geometric progression.

        See Also
        --------
        logspace : Similar to geomspace, but with endpoints specified using log
                   and base.
        linspace : Similar to geomspace, but with arithmetic instead of geometric
                   progression.
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples.

        Examples
        --------
        >>> import mygrad as mg
        >>> import numpy as np
        >>> mg.geomspace(1, 1000, num=4)
        Tensor([    1.,    10.,   100.,  1000.])
        >>> mg.geomspace(1, 1000, num=3, endpoint=False)
        Tensor([   1.,   10.,  100.])
        >>> mg.geomspace(1, 1000, num=4, endpoint=False)
        Tensor([   1.        ,    5.62341325,   31.6227766 ,  177.827941  ])
        >>> mg.geomspace(1, 256, num=9)
        Tensor([   1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.])

        Note that the above may not produce exact integers:

        >>> mg.geomspace(1, 256, num=9, dtype=int)
        Tensor([  1,   2,   4,   7,  16,  32,  63, 127, 256])
        >>> np.around(mg.geomspace(1, 256, num=9).data).astype(int)
        array([  1,   2,   4,   8,  16,  32,  64, 128, 256])

        Negative, decreasing, and complex inputs are allowed:

        >>> mg.geomspace(1000, 1, num=4)
        Tensor([ 1000.,   100.,    10.,     1.])
        >>> mg.geomspace(-1000, -1, num=4)
        Tensor([-1000.,  -100.,   -10.,    -1.])
    """
    return Tensor(np.geomspace(start, stop, num, include_endpoint, dtype), constant=constant)
