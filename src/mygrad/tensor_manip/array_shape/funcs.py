from typing import Callable, Optional, Tuple, Union, List, cast, TypeVar, overload

from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike, Shape

from .ops import *

__all__ = [
    "reshape",
    "squeeze",
    "ravel",
    "expand_dims",
    "broadcast_to",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
]

_T = TypeVar("_T")


@implements_numpy_override()
def reshape(
    a: ArrayLike, newshape: Union[int, Shape], *, constant: Optional[bool] = None
) -> Tensor:
    """Returns a tensor with a new shape, without changing its data.

    This docstring was adapted from ``numpy.reshape``

    Parameters
    ----------
    a : ArrayLike
        The tensor to be reshaped

    newshape : Union[int, Tuple[int, ...]]
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D tensor of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the tensor and remaining dimensions.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        ``a`` with its shape changed permuted.  A new tensor is returned.

    Notes
    -----
    ``reshape`` utilizes C-ordering, meaning that it reads & writes elements using
    C-like index ordering; the last axis index changing fastest, and, proceeding
    in reverse order, the first axis index changing slowest.

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.Tensor([[1,2,3], [4,5,6]])
    >>> mg.reshape(a, 6)
    Tensor([1, 2, 3, 4, 5, 6])

    >>> mg.reshape(a, (3,-1))   # the unspecified value is inferred to be 2
    Tensor([[1, 2],
            [3, 4],
            [5, 6]])"""
    return Tensor._op(Reshape, a, op_args=(newshape,), constant=constant)


@implements_numpy_override()
def squeeze(
    a: ArrayLike,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    constant: Optional[bool] = None
) -> Tensor:
    """
    Remove single-dimensional entries from the shape of a tensor.

    This docstring was adapted from ``numpy.squeeze``

    Parameters
    ----------
    a : ArrayLike
        The tensor to be reshaped

    axis : Optional[int, Tuple[int, ...]]
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Raises
    ------
    ValueError
        If ``axis`` is not ``None``, and an axis being squeezed is not of length 1

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.Tensor([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> mg.squeeze(x).shape
    (3,)
    >>> mg.squeeze(x, axis=0).shape
    (3, 1)
    >>> mg.squeeze(x, axis=1).shape
    Traceback (most recent call last) -> Tensor:
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> mg.squeeze(x, axis=2).shape
    (1, 3)"""
    return Tensor._op(Squeeze, a, op_args=(axis,), constant=constant)


@implements_numpy_override()
def ravel(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """
    Flattens contents of a tensor into a contiguous 1-D array.  A copy is made only if needed.

    This docstring was adapted from ``numpy.ravel``.

    Parameters
    ----------
    a : ArrayLike
        The tensor to be flattened

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    ``ravel`` utilizes C-ordering, meaning that it reads & writes elements using
    C-like index ordering; the last axis index changing fastest, and, proceeding
    in reverse order, the first axis index changing slowest.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.Tensor([[1, 2],
    ...                [3, 4]])
    >>> mg.ravel(x)
    Tensor([1, 2, 3, 4])
    """
    return Tensor._op(Ravel, a, constant=constant)


@implements_numpy_override()
def expand_dims(a: ArrayLike, axis: int, *, constant: Optional[bool] = None) -> Tensor:
    """
    Expand the dimensions of a tensor by adding a new axis.

    This docstring was adapted from ``numpy.expand_dims``.

    Parameters
    ----------
    a : ArrayLike
        The tensor to be expanded

    axis : int
        The position of the new axis in the expanded array shape.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.Tensor([1, 2])
    >>> x.shape
    (2,)
    >>> y = mg.expand_dims(x, 1)
    >>> y.shape
    (2, 1)
    >>> z = mg.expand_dims(y, 0)
    >>> z.shape
    (1, 2, 1)
    """
    return Tensor._op(ExpandDims, a, op_args=(axis,), constant=constant)


@implements_numpy_override()
def broadcast_to(
    a: ArrayLike, shape: Shape, *, constant: Optional[bool] = None
) -> Tensor:
    """
    Broadcast a tensor to a new shape.

    This docstring was adapted from ``numpy.broadcast_to``.

    Parameters
    ----------
    a : ArrayLike
        The tensor to be broadcasted

    shape: Tuple[int, ...]
        The shape of the broadcasted tensor. This shape
        should be broadcast-compatible with the original
        shape.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Raises
    ------
    ValueError
        If the array is not compatible with the new shape
        according to Numpy's broadcasting rules.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.Tensor([1, 2, 3])
    >>> mg.broadcast_to(x, (3,3))
    Tensor([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]])
    >>> mg.broadcast_to(x, (4,4))
    Traceback (most recent call last) -> Tensor:
    ...
    ValueError: operands could not be broadcast together with remapped
    shapes [original->remapped]: (3,) and requested shape (4,4)
    """
    return Tensor._op(BroadcastTo, a, op_args=(shape,), constant=constant)


def _dispatch_atleast_kd(func: Callable[..., _T], Op, *tensors, k: int, constant) -> _T:
    if len(tensors) == 1:
        (t,) = tensors
        if (
            isinstance(t, Tensor)
            and t.ndim >= k
            and (constant is None or t.constant is constant)
        ):
            # return tensor unchanged
            return cast(_T, t)
        return cast(_T, Tensor._op(Op, t, constant=constant))
    else:
        out = [func(t, constant=constant) for t in tensors]
        return cast(_T, out)


@overload
def atleast_1d(tensors: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    ...


@overload
def atleast_1d(*tensors: ArrayLike, constant: Optional[bool] = None) -> List[Tensor]:
    ...


@implements_numpy_override()
def atleast_1d(
    *tensors: ArrayLike, constant: Optional[bool] = None
) -> Union[Tensor, List[Tensor]]:
    """
    Convert inputs to tensors with at least one dimension.

    Scalar inputs are converted to 1-dimensional tensors, whilst
    higher-dimensional inputs are preserved.

    This docstring was adapted from ``numpy.atleast_1d``.

    Parameters
    ----------
    tens1, tens2, ... : ArrayLike
        One or more input tensors.

    Returns
    -------
    ret : Tensor
        A tensor, or list of tensors, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.atleast_1d(1.0)
    array([1.])

    >>> x = mg.arange(9.0).reshape(3,3)
    >>> np.atleast_1d(x)
    Tensor([[0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]])
    >>> mg.atleast_1d(x) is x
    True

    >>> mg.atleast_1d(1, [3, 4])
    [Tensor([1]), Tensor([3, 4])]

    ``numpy.atleast_1d`` will dispatch appropriately on tensors.

    >>> x = mg.tensor(2.)
    >>> np.atleast_1d(x)
    Tensor([2.])

    >>> np.atleast_1d(x).backward()
    >>> x.grad
    array(1.)

    If any argument to ``numpy.atleast_1d`` is a Tensor, ``mygrad.atleast_1d``
    will be dispatched on all of the arguments.

    >>> np.atleast_1d(x, 1.)
    [Tensor([2.]), Tensor([1.])]
    """
    return _dispatch_atleast_kd(atleast_1d, AtLeast1D, *tensors, k=1, constant=constant)


@overload
def atleast_2d(tensors: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    ...


@overload
def atleast_2d(*tensors: ArrayLike, constant: Optional[bool] = None) -> List[Tensor]:
    ...


@implements_numpy_override()
def atleast_2d(
    *tensors: ArrayLike, constant: Optional[bool] = None
) -> Union[Tensor, List[Tensor]]:
    """
    Convert inputs to tensors with at least one dimension.

    Scalar inputs are converted to 2-dimensional tensors, whilst
    higher-dimensional inputs are preserved.

    This docstring was adapted from ``numpy.atleast_2d``.

    Parameters
    ----------
    tens1, tens2, ... : ArrayLike
        One or more input tensors.

    Returns
    -------
    ret : Tensor
        A tensor, or list of tensors, each with ``a.ndim >= 2``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.atleast_2d(3.0)
    Tensor([[3.]])

    >>> x = mg.arange(3.0)
    >>> mg.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> mg.atleast_2d(x).base is x
    True

    >>> mg.atleast_2d(1, [1, 2], [[1, 2]])
    [Tensor([[1]]), Tensor([[1, 2]]), Tensor([[1, 2]])]

    ``numpy.atleast_2d`` will dispatch appropriately on tensors.

    >>> x = mg.tensor(2.)
    >>> np.atleast_2d(x)
    Tensor([[2.]])

    >>> np.atleast_2d(x).backward()
    >>> x.grad
    array(1.)

    If any argument to ``numpy.atleast_2d`` is a Tensor, ``mygrad.atleast_1d``
    will be dispatched on all of the arguments.

    >>> np.atleast_2d(x, 1.)
    [Tensor([[2.]]), Tensor([[1.]])]
    """
    return _dispatch_atleast_kd(atleast_2d, AtLeast2D, *tensors, k=2, constant=constant)


@overload
def atleast_3d(tensors: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    ...


@overload
def atleast_3d(*tensors: ArrayLike, constant: Optional[bool] = None) -> List[Tensor]:
    ...


@implements_numpy_override()
def atleast_3d(
    *tensors: ArrayLike, constant: Optional[bool] = None
) -> Union[Tensor, List[Tensor]]:
    """
    Convert inputs to tensors with at least one dimension.

    Scalar inputs are converted to 2-dimensional tensors, whilst
    higher-dimensional inputs are preserved.

    This docstring was adapted from ``numpy.atleast_3d``.

    Parameters
    ----------
    tens1, tens2, ... : ArrayLike
        One or more input tensors.

    Returns
    -------
    ret : Tensor
        A tensor, or list of tensors, each with ``a.ndim >= 2``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.atleast_3d(3.0)
    Tensor([[[3.]]])

    >>> x = mg.arange(3.0)
    >>> mg.atleast_3d(x)
    array([[0., 1., 2.]])
    >>> mg.atleast_3d(x).base is x
    True

    >>> mg.atleast_3d(1, [[1, 2]], [[[[1, 2]]]])
    [Tensor([[[1]]]), Tensor([[[1, 2]]]), Tensor([[[[1, 2]]]])]

    ``numpy.atleast_3d`` will dispatch appropriately on tensors.

    >>> x = mg.tensor(2.)
    >>> np.atleast_3d(x)
    Tensor([[[2.]]])

    >>> np.atleast_3d(x).backward()
    >>> x.grad
    array(1.)

    If any argument to ``numpy.atleast_3d`` is a Tensor, ``mygrad.atleast_1d``
    will be dispatched on all of the arguments.

    >>> np.atleast_3d(x, 1.)
    [Tensor([[[2.]]]), Tensor([[[1.]]])]
    """
    return _dispatch_atleast_kd(atleast_3d, AtLeast3D, *tensors, k=2, constant=constant)
