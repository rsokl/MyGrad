from typing import Optional, Union

from numpy import ndarray

from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike, DTypeLikeReals, Mask
from mygrad.ufuncs._ufunc_creators import ufunc_creator

from .ops import (
    Add,
    AddSequence,
    Divide,
    Multiply,
    MultiplySequence,
    Negative,
    Positive,
    Power,
    Reciprocal,
    Square,
    Subtract,
)

__all__ = [
    "add",
    "add_sequence",
    "divide",
    "multiply",
    "multiply_sequence",
    "negative",
    "positive",
    "power",
    "reciprocal",
    "square",
    "subtract",
    "true_divide",
]


@ufunc_creator(Add)
def add(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Add the arguments element-wise.

    This docstring was adapted from that of numpy.add [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output). Non-tensor array-likes are
        treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[ndarray, Tensor]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    add : Tensor
        The sum of `x1` and `x2`, element-wise.

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of tensor broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.add.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.add(1.0, 4.0)
    Tensor(5.0)
    >>> x1 = mg.tensor([[0., 1., 2.],
    ...                 [3., 4., 5.],
    ...                 [6., 7., 8.]])
    >>> x2 = mg.tensor([0., 1., 2.])
    >>> mg.add(x1, x2)
    Tensor([[  0.,   2.,   4.],
            [  3.,   5.,   7.],
            [  6.,   8.,  10.]])
    """
    ...


@ufunc_creator(Subtract)
def subtract(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Subtract the arguments element-wise.

    This docstring was adapted from that of numpy.subtract [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        The arrays to be subtracted from each other.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output). Non-tensor array-likes are
        treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[ndarray, Tensor]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    subtract : Tensor
        The difference of `x1` and `x2`, element-wise.

    Notes
    -----
    Equivalent to ``x1 - x2`` in terms of tensor broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.subtract.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.subtract(1.0, 4.0)
    Tensor(-3.0)

    >>> x1 = mg.tensor([[0., 1., 2.],
    ...                 [3., 4., 5.],
    ...                 [6., 7., 8.]])
    >>> x2 = mg.tensor([0., 1., 2.])
    >>> mg.subtract(x1, x2)
    Tensor([[ 0.,  0.,  0.],
            [ 3.,  3.,  3.],
            [ 6.,  6.,  6.]])
    """
    ...


@ufunc_creator(Multiply)
def multiply(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Multiply the arguments element-wise.

    This docstring was adapted from that of numpy.multiply [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        Input arrays to be multiplied.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output). Non-tensor array-likes
        are treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[ndarray, Tensor]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    multiply : Tensor
        The product of `x1` and `x2`, element-wise.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of tensor broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.multiply.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.multiply(2.0, 4.0)
    Tensor(8.0)

    >>> x1 = mg.tensor([[0., 1., 2.],
    ...                 [3., 4., 5.],
    ...                 [6., 7., 8.]])
    >>> x2 = mg.tensor([0., 1., 2.])
    >>> mg.multiply(x1, x2)
    Tensor([[  0.,   1.,   4.],
            [  0.,   4.,  10.],
            [  0.,   7.,  16.]])
    """
    ...


@ufunc_creator(Divide)
def true_divide(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Divide the arguments element-wise.

    This docstring was adapted from that of numpy.true_divide [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Dividend array.

    x2 : ArrayLike
        Divisor array.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output). Non-tensor array-likes
        are treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[ndarray, Tensor]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    true_divide : Tensor
        The quotient of `x1` with `x2`, element-wise.

    Notes
    -----
    In Python, ``//`` is the floor division operator and ``/`` the
    true division operator.  The ``true_divide(x1, x2)`` function is
    equivalent to true division in Python.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.arange(5)
    >>> mg.true_divide(x, 4)
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    >>> x/4
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    >>> x//4
    array([0, 0, 0, 0, 1])
    """
    ...


divide = true_divide


@ufunc_creator(Power)
def power(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """First tensor elements raised to powers from second tensor, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in
    `x2`.  `x1` and `x2` must be broadcastable to the same shape. Note that an
    integer type raised to a negative integer power will raise a ValueError.

    This docstring was adapted from that of numpy.power [1]_

    Parameters
    ----------
    x1 : ArrayLike
        The bases.

    x2 : ArrayLike
        The exponents.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output). Non-tensor array-likes
        are treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[ndarray, Tensor]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    power : Tensor
        The combination of `x1` and `x2`, element-wise.

    See Also
    --------
    float_power : power function that promotes integers to float

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.power.html

    Examples
    --------
    Cube each element in a list.

    >>> import mygrad as mg
    >>> x1 = range(6)
    >>> x1
    [0, 1, 2, 3, 4, 5]
    >>> mg.power(x1, 3)
    Tensor([  0,   1,   8,  27,  64, 125])

    Raise the bases to different exponents.

    >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    >>> mg.power(x1, x2)
    Tensor([  0.,   1.,   8.,  27.,  16.,   5.])

    The effect of broadcasting.

    >>> x2 = mg.tensor([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> x2
    Tensor([[1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1]])
    >>> mg.power(x1, x2)
    Tensor([[ 0,  1,  8, 27, 16,  5],
            [ 0,  1,  8, 27, 16,  5]])
    """
    ...


@ufunc_creator(Negative)
def negative(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Negates the tensor element-wise.

    This docstring was adapted from that of numpy.negative [1]_

    Parameters
    ----------
    x : ArrayLike or scalar
        Input tensor.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    negative : Tensor
        The combination of `x1` and `x2`, element-wise.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.negative.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.negative([1.,-1.])
    Tensor([-1.,  1.])
    """
    ...


@ufunc_creator(Positive)
def positive(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Returns a copy of the tensor.

    This docstring was adapted from that of numpy.positive [1]_

    Parameters
    ----------
    x : ArrayLike
        Input array.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    positive : Tensor

    Notes
    -----
    Equivalent to `x.copy()`, but only defined for types that support
    arithmetic.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.positive.html
    """
    ...


@ufunc_creator(Reciprocal)
def reciprocal(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return the reciprocal of the argument element-wise.

    This docstring was adapted from that of numpy.reciprocal [1]_

    Parameters
    ----------
    x : ArrayLike
        Input array.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    reciprocal : Tensor

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.reciprocal(2.)
    Tensor(0.5)
    >>> mg.reciprocal([1, 2., 3.33])
    Tensor([ 1.       ,  0.5      ,  0.3003003])
    """
    ...


@ufunc_creator(Square)
def square(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return the square of the argument element-wise.

    This docstring was adapted from that of numpy.square [1]_

    Parameters
    ----------
    x : ArrayLike
        Input data.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    Returns
    -------
    square : Tensor

    See Also
    --------
    sqrt
    power

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.square.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.square([100., 1000.])
    array([10.,  100.])
    """
    ...


def multiply_sequence(*variables: ArrayLike, constant: Optional[bool] = None) -> Tensor:
    """``f(a, b, ...) -> a * b * ...``

    Multiply a sequence of tensors.

    Parameters
    ----------
    variables : ArrayLike
        A sequence of broadcast-compatible tensors. Non-tensor array-likes are
        treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    It is more efficient to back-propagate through this
    function than it is through a computational graph
    with N-1 corresponding multiplication operations.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.tensor([1. , 2.])
    >>> y = mg.tensor([-1.])
    >>> z = mg.tensor([[1.]])
    >>> out = mg.multiply_sequence(x, y, z); out
        Tensor([[-1., -2.]])

    >>> out.backward()
    >>> x.grad
    array([-1., -1.])
    >>> y.grad
    array([3.])
    >>> z.grad
    array([[-3.]])
    """
    if len(variables) < 2:
        raise ValueError(
            f"`multiply_sequence` requires at least two inputs, got {len(variables)} inputs"
        )
    return Tensor._op(MultiplySequence, *variables, constant=constant)


def add_sequence(*variables: ArrayLike, constant: Optional[bool] = None) -> Tensor:
    """``f(a, b, ...) -> a + b + ...``

    Add a sequence of tensors.

    Parameters
    ----------
    variables : ArrayLike
        A sequence of broadcast-compatible tensors. Non-tensor array-likes are
        treated as constants.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    It is more efficient to back-propagate through this
    function than it is through a computational graph
    with N-1 corresponding addition operations.


    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.tensor([1. , 2.])
    >>> y = mg.tensor([-1.])
    >>> z = mg.tensor([[1.]])
    >>> out = mg.add_sequence(x, y, z); out
        Tensor([[1., 2.]])

    >>> out.backward()
    >>> x.grad
    array([1., 1.])
    >>> y.grad
    array([2.])
    >>> z.grad
    array([[2.]])
    """
    if len(variables) < 2:
        raise ValueError(
            f"`add_sequence` requires at least two inputs, got {len(variables)} inputs"
        )
    return Tensor._op(AddSequence, *variables, constant=constant)
