from typing import Optional, Sequence, Union

import numpy as np

from mygrad.tensor_base import Tensor, _resolve_constant, implements_numpy_override
from mygrad.typing import ArrayLike, DTypeLikeReals, Real

Shape = Union[Sequence[int], int]


def _anything_but_tensor(x):
    if isinstance(x, Tensor):
        x = x.data
    return x


__all__ = [
    "arange",
    "empty",
    "empty_like",
    "eye",
    "geomspace",
    "identity",
    "linspace",
    "logspace",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "zeros",
    "zeros_like",
]


def empty(
    shape: Shape, dtype: DTypeLikeReals = np.float32, *, constant: Optional[bool] = None
) -> Tensor:
    """Return a new Tensor of the given shape and type, without initializing entries.

    This docstring was adapted from ``numpy.empty`` [1]_

    Parameters
    ----------
    shape : Union[int, Tuple[int]]
        The shape of the empty array.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A tensor of uninitialized data of the given shape and dtype.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.empty.html

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
    return Tensor(np.empty(shape=shape, dtype=dtype), constant=constant, copy=False)


@implements_numpy_override
def empty_like(
    other: ArrayLike,
    dtype: Optional[DTypeLikeReals] = None,
    shape: Optional[Union[int, Sequence[int]]] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a new Tensor of the same shape and type as the given array.

    This docstring was adapted from ``numpy.empty_like`` [1]_

    Parameters
    ----------
    other : ArrayLike
        The Tensor or array whose shape and datatype should be mirrored.

    dtype : Optional[DTypeLikeReals]
        Override the data type of the returned Tensor with this value, or None to not override.

    shape : Optional[Union[int, Sequence[int]]]
        If specified, overrides the shape of the result

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation. If ``None`` then:

        Inferred from ``other``, if other is a tensor
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

    Returns
    -------
    Tensor
        A tensor of uninitialized data whose shape and type match `other`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html

    See Also
    --------
    empty : Return a new Tensor of the given shape and type, without initializing entries.
    ones : Return a new tensor setting values to one.
    zeros : Return a new tensor setting values to zero.
    full : Return a new tensor of given shape filled with value.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.arange(4).reshape(2, 2)
    >>> mg.empty_like(x, constant=True)
    Tensor([[ -9.74499359e+001,   6.69583040e-309],
            [  2.13182611e-314,   3.06959433e-309]])         #random

    >>> mg.empty_like(x, dtype=int)
    Tensor([[-1073741821, -1067949133],
            [  496041986,    19249760]])                     #random
    """
    constant = _resolve_constant(other, constant=constant)
    return Tensor(
        np.empty_like(_anything_but_tensor(other), dtype=dtype, shape=shape),
        constant=constant,
        copy=False,
    )


def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: DTypeLikeReals = float,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a 2D Tensor with ones on the diagonal and zeros elsewhere.

    This docstring was adapted from ``numpy.eye`` [1]_

    Parameters
    ----------
    N : int
        The number of rows in the output Tensor.

    M : int, optional (default=None)
        The number of columns in the output, or None to match `rows`.

    k : int, optional (default=0)
        The index of the diagonal. 0 is the main diagonal; a positive value is the upper
        diagonal, while a negative value refers to the lower diagonal.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.eye.html

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
    return Tensor(
        np.eye(N, M=M, k=k, dtype=dtype),
        constant=constant,
        copy=False,
    )


def identity(
    n: int, dtype: DTypeLikeReals = float, *, constant: Optional[bool] = None
) -> Tensor:
    """Return the identity Tensor; a square Tensor with 1s on the main diagonal and 0s elsewhere.

    This docstring was adapted from ``numpy.identity`` [1]_

    Parameters
    ----------
    n : int
        The number of rows and columns in the output Tensor.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A square Tensor whose main diagonal is 1 and all other elements are 0.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.identity.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.identity(3)
    Tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    """
    return Tensor(np.identity(n, dtype=dtype), constant=constant, copy=False)


def ones(
    shape: Shape, dtype: DTypeLikeReals = np.float32, *, constant: Optional[bool] = None
) -> Tensor:
    """
    Return a Tensor of the given shape and type, filled with ones.

    This docstring was adapted from ``numpy.ones`` [1]_

    Parameters
    ----------
    shape : Union[int, Tuple[int]]
        The shape of the output Tensor.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A Tensor of ones with the given shape and data type.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.ones.html

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
    return Tensor(np.ones(shape, dtype=dtype), constant=constant, copy=False)


@implements_numpy_override
def ones_like(
    other: ArrayLike,
    dtype: Optional[DTypeLikeReals] = None,
    shape: Optional[Union[int, Sequence[int]]] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    Return a Tensor of the same shape and type as the given, filled with ones.

    This docstring was adapted from ``numpy.ones_like`` [1]_

    Parameters
    ----------
    other : array_like
        The Tensor or array whose shape and datatype should be mirrored.

    dtype : Optional[DTypeLikeReals]
        Override the data type of the returned Tensor with this value, or None to not override.

    shape : Optional[Union[int, Sequence[int]]]
        If specified, overrides the shape of the result

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation. If ``None`` then:

        Inferred from ``other``, if other is a tensor
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.


    Returns
    -------
    Tensor
        A Tensor of ones whose shape and data type match `other`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html

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
    constant = _resolve_constant(other, constant=constant)

    return Tensor(
        np.ones_like(_anything_but_tensor(other), dtype=dtype, shape=shape),
        constant=constant,
        copy=False,
    )


def zeros(
    shape: Shape, dtype: DTypeLikeReals = np.float32, *, constant: Optional[bool] = None
) -> Tensor:
    """
    Return a Tensor of the given shape and type, filled with zeros.

    This docstring was adapted from ``numpy.zeros`` [1]_

    Parameters
    ----------
    shape : Union[int, Tuple[int]]
        The shape of the output Tensor.

    dtype : data-type, optional (default=numpy.float32)
        The data type of the output Tensor.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A Tensor of zeros with the given shape and data type.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.zeros.html

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
    return Tensor(np.zeros(shape, dtype), constant=constant, copy=False)


@implements_numpy_override
def zeros_like(
    other: ArrayLike,
    dtype: Optional[DTypeLikeReals] = None,
    shape: Optional[Union[int, Shape]] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    Return a Tensor of the same shape and type as the given, filled with zeros.

    This docstring was adapted from ``numpy.zeros_like`` [1]_

    Parameters
    ----------
    other : ArrayLike
        The Tensor or array whose shape and datatype should be mirrored.

    dtype : Optional[DTypeLikeReals]
        Override the data type of the returned Tensor with this value, or None to not override.

    shape : Optional[int, Sequence[int]]
        If specified, overrides the shape of the result

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation. If ``None`` then:

        Inferred from ``other``, if other is a tensor
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A Tensor of zeros whose shape and data type match `other`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html

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
    constant = _resolve_constant(other, constant=constant)
    return Tensor(
        np.zeros_like(_anything_but_tensor(other), dtype=dtype, shape=shape),
        constant=constant,
        copy=False,
    )


def full(
    shape: Shape,
    fill_value: ArrayLike,
    dtype: Optional[DTypeLikeReals] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    Return a Tensor of the given shape and type, filled with `fill_value`.

    This docstring was adapted from ``numpy.full`` [1]_

    Parameters
    ----------
    shape : Union[int, Iterable[int]]
        The shape of the output Tensor.

    fill_value : ArrayLike
        The value with which to fill the output Tensor. Note that this function
        is not differentiable – the resulting tensor will not backprop through
        `fill_value`.

        The value with which to fill the output Tensor.

    dtype : Optional[DTypeLikeReals]
        The data type of the output Tensor, or None to match `fill_value`..

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A Tensor of `fill_value` with the given shape and dtype.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.full.html

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
    return Tensor(
        np.full(shape, fill_value=fill_value, dtype=dtype),
        constant=constant,
        copy=False,
    )


@implements_numpy_override
def full_like(
    other: ArrayLike,
    fill_value: Real,
    dtype: Optional[DTypeLikeReals] = None,
    shape: Optional[Union[int, Shape]] = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a Tensor of the same shape and type as the given, filled with `fill_value`.

    This docstring was adapted from ``numpy.full_like`` [1]_

    Parameters
    ----------
    other : ArrayLike
        The tensor or array whose shape and datatype should be mirrored.

    fill_value : Real
        The value with which to fill the output Tensor.

    dtype : Optional[DTypeLikeReals]
        Override the data type of the returned Tensor with this value, or None to not override.

    shape : Optional[int, Sequence[int]]
        If specified, overrides the shape of the result

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation. If ``None`` then:

        Inferred from ``other``, if other is a tensor
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

    Returns
    -------
    Tensor
        A Tensor of `fill_value` whose shape and data type match `other`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.full_like.html

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
    constant = _resolve_constant(other, constant=constant)

    return Tensor(
        np.full_like(
            _anything_but_tensor(other),
            fill_value=_anything_but_tensor(fill_value),
            dtype=dtype,
            shape=shape,
        ),
        constant=constant,
        copy=False,
    )


def arange(
    start: Real,
    stop: Real = None,
    step: int = None,
    dtype: Optional[DTypeLikeReals] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a Tensor with evenly-spaced values within a given interval.

    Values are generated within [start, stop). Note that for non-integer steps, results may be
    inconsistent; you are better off using `linspace` instead.

    This docstring was adapted from ``numpy.arange`` [1]_

    Parameters
    ----------
    start : Real, optional, default=0
        The start of the interval, inclusive.

    stop : Real
        The end of the interval, exclusive.

    step : int, optional (default=1)
        The spacing between successive values.

    dtype : Optional[DTypeLikeReals]
        The data type of the output Tensor, or None to infer from the inputs.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        A Tensor of evenly-spaced values in [start, end).

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arange.html

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
    if stop is None:
        arr = np.arange(start, step=step, dtype=dtype)
    else:
        arr = np.arange(start, stop, step=step, dtype=dtype)

    return Tensor(arr, constant=constant, copy=False)


def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    dtype: Optional[DTypeLikeReals] = None,
    axis: int = 0,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a Tensor with evenly-spaced numbers over a specified interval.

    Values are generated within [start, stop], with the endpoint optionally excluded.

    This docstring was adapted from ``numpy.linspace`` [1]_

    Parameters
    ----------
    start : ArrayLike
        The starting value of the sequence, inclusive.

    stop : ArrayLike
        The ending value of the sequence, inclusive unless `include_endpoint` is False.

    num : int, optional (default=50)
        The number of values to generate. Must be non-negative.

    endpoint : bool, optional (default=True)
        Whether to include the endpoint in the Tensor. Note that if False, the step size changes
        to accommodate the sequence excluding the endpoint.

    dtype : Optional[DTypeLikeReals]
        The data type of the output Tensor, or None to infer from the inputs.

    axis : int, optional (default=0)
        The axis in the result to store the samples - for array-like start/stop.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

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
    return Tensor(
        np.linspace(
            start,
            stop,
            num,
            endpoint=endpoint,
            dtype=dtype,
            axis=axis,
        ),
        constant=constant,
        copy=False,
    )


def logspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    base: Real = 10,
    dtype: Optional[DTypeLikeReals] = None,
    axis: int = 0,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a Tensor with evenly-spaced numbers over a specified interval on a log scale.
    This is not a differentiable function - it does not propagate gradients to its inputs.

    In linear space, values are generated within [base**start, base**stop], with the endpoint
    optionally excluded.

    This docstring was adapted from ``numpy.logspace`` [1]_

    Parameters
    ----------
    start : ArrayLike
        The starting value of the sequence, inclusive; start at `base ** start`.

    stop : ArrayLike
        The ending value of the sequence, inclusive unless `include_endpoint` is False; end at
        `base ** stop`.

    num : int, optional (default=50)
        The number of values to generate. Must be non-negative.

    endpoint : bool, optional (default=True)
        Whether to include the endpoint in the Tensor. Note that if False, the step size changes
        to accommodate the sequence excluding the endpoint.

    base : Real, optional (default=10)
        The base of the log space.

    dtype : Optional[DTypeLikeReals]
        The data type of the output Tensor, or None to infer from the inputs.

    axis : int, optional (default=0)
        The axis in the result to store the samples - for array-like start/stop.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.
    geomspace : Similar to logspace, but with endpoints specified directly.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.logspace.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.logspace(2.0, 3.0, num=4)
    Tensor([  100.        ,   215.443469  ,   464.15888336,  1000.        ])
    >>> mg.logspace(2.0, 3.0, num=4, endpoint=False)
    Tensor([ 100.        ,  177.827941  ,  316.22776602,  562.34132519])
    >>> mg.logspace(2.0, 3.0, num=4, base=2.0)
    Tensor([ 4.        ,  5.0396842 ,  6.34960421,  8.        ])

    """
    return Tensor(
        np.logspace(
            start=start,
            stop=stop,
            num=num,
            endpoint=endpoint,
            base=base,
            dtype=dtype,
            axis=axis,
        ),
        constant=constant,
        copy=False,
    )


def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    dtype=None,
    axis=0,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return a Tensor with evenly-spaced values in a geometric progression.

    Each output sample is a constant multiple of the previous output.

    This docstring was adapted from ``numpy.geomspace`` [1]_

    Parameters
    ----------
    start : ArrayLike
        The starting value of the output.

    stop : ArrayLike
        The ending value of the sequence, inclusive unless `endpoint` is false.

    num : int, optional (default=50)
        The number of values to generate. Must be non-negative.

    endpoint : bool, optional (default=True)
        Whether to include the endpoint in the Tensor. Note that if False, the step size changes
        to accommodate the sequence excluding the endpoint.

    dtype : Optional[DTypeLikeReals]
        The data type of the output Tensor, or None to infer from the inputs.

    axis : int, optional (default=0)
        The axis in the result to store the samples - for array-like start/stop.

    constant : Optional[bool]
        If ``True``, this tensor is a constant, and thus does not facilitate
        back propagation.

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html

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

    Negative, and decreasing inputs are allowed:

    >>> mg.geomspace(1000, 1, num=4)
    Tensor([ 1000.,   100.,    10.,     1.])
    >>> mg.geomspace(-1000, -1, num=4)
    Tensor([-1000.,  -100.,   -10.,    -1.])
    """
    return Tensor(
        np.geomspace(
            start=start,
            stop=stop,
            num=num,
            endpoint=endpoint,
            dtype=dtype,
            axis=axis,
        ),
        constant=constant,
        copy=False,
    )
