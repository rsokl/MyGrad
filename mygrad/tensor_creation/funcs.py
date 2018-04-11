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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A tensor of uninitialized data of the given shape and dtype.
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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A tensor of uninitialized data whose shape and type match `other`.
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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A tensor whose elements are 0, except for the :math:`k`-th diagonal, whose values are 1.
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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A square Tensor whose main diagonal is 1 and all other elements are 0.
    """
    return Tensor(np.identity(n, dtype), constant=constant)
    

def ones(shape, dtype=np.float32, constant=False):
    """ Return a Tensor of the given shape and type, filled with ones.

        Parameters
        ----------
        shape : Union[int, Tuple[int]]
            The shape of the output Tensor.

        dtype : data-type, optional (default=numpy.float32)
            The data type of the output Tensor.

        constant : bool, optional (default=False)
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of ones with the given shape and data type.
    """
    return Tensor(np.ones(shape, dtype), constant=constant)


def ones_like(other, dtype=None, constant=False):
    """ Return a Tensor of the same shape and type as the given, filled with ones.

        Parameters
        ----------
        other : Union[Tensor, ArrayLike]
            The Tensor or array whose shape and datatype should be mirrored.

        dtype : data-type, optional (default=None)
            Override the data type of the returned Tensor with this value, or None to not override.

        constant : bool, optional (default=False)
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of ones whose shape and data type match `other`.
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.ones_like(other, dtype), constant=constant)


def zeros(shape, dtype=np.float32, constant=False):
    """ Return a Tensor of the given shape and type, filled with zeros.

        Parameters
        ----------
        shape : Union[int, Tuple[int]]
            The shape of the output Tensor.

        dtype : data-type, optional (default=numpy.float32)
            The data type of the output Tensor.

        constant : bool, optional (default=False)
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of zeros with the given shape and data type.
    """
    return Tensor(np.zeros(shape, dtype), constant=constant)


def zeros_like(other, dtype=None, constant=False):
    """ Return a Tensor of the same shape and type as the given, filled with zeros.

        Parameters
        ----------
        other : Union[Tensor, ArrayLike]
            The Tensor or array whose shape and datatype should be mirrored.

        dtype : data-type, optional (default=None)
            Override the data type of the returned Tensor with this value, or None to not override.

        constant : bool, optional (default=False)
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of zeros whose shape and data type match `other`.
    """
    if isinstance(other, Tensor):
        other = other.data

    return Tensor(np.zeros_like(other, dtype), constant=constant)


def full(shape, fill_value, dtype=None, constant=False):
    """ Return a Tensor of the given shape and type, filled with `fill_value`.

        Parameters
        ----------
        shape : Union[int, Tuple[int]]
            The shape of the output Tensor.

        fill_value : Real
            The value with which to fill the output Tensor.

        dtype : data-type, optional (default=None)
            The data type of the output Tensor, or None to match `fill_value`..

        constant : bool, optional (default=False)
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of `fill_value` with the given shape and dtype.
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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of `fill_value` whose shape and data type match `other`.
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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of evenly-spaced values in [start, end).
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
            Whether the output Tensor is a constant Tensor.

        Returns
        -------
        Tensor
            A Tensor of `num` evenly-spaced values in [start, stop] or [start, stop), depending on
            `include_endpoint`.
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
            Whether the output Tensor is a constant Tensor.

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

        Returns
        -------
        Tensor
            A Tensor of `num` samples, evenly-spaced in a geometric progression.
    """
    return Tensor(np.geomspace(start, stop, num, include_endpoint, dtype), constant=constant)
