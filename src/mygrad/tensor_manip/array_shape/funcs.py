from mygrad.tensor_base import Tensor

from .ops import BroadcastTo, ExpandDims, Ravel, Reshape, Squeeze

__all__ = ["reshape", "squeeze", "ravel", "expand_dims", "broadcast_to"]


def reshape(a, newshape, constant=False):
    """ Returns a tensor with a new shape, without changing its data.

    This docstring was adapted from ``numpy.reshape``

    Parameters
    ----------
    a : array_like
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


def squeeze(a, axis=None, constant=False):
    """
    Remove single-dimensional entries from the shape of a tensor.

    This docstring was adapted from ``numpy.squeeze``

    Parameters
    ----------
    a : array_like
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
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> mg.squeeze(x, axis=2).shape
    (1, 3)"""
    return Tensor._op(Squeeze, a, op_args=(axis,), constant=constant)


def ravel(a, constant=False):
    """
    Flattens contents of a tensor into a contiguous 1-D array.  A copy is made only if needed.

    This docstring was adapted from ``numpy.ravel``.

    Parameters
    ----------
    a : array_like
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


def expand_dims(a, axis, constant=False):
    """
    Expand the dimensions of a tensor by adding a new axis.

    This docstring was adapted from ``numpy.expand_dims``.

    Parameters
    ----------
    a : array_like
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


def broadcast_to(a, shape, constant=False):
    """
    Broadcast a tensor to a new shape.

    This docstring was adapted from ``numpy.broadcast_to``.

    Parameters
    ----------
    a : array_like
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
    Traceback (most recent call last):
    ...
    ValueError: operands could not be broadcast together with remapped
    shapes [original->remapped]: (3,) and requested shape (4,4)
    """
    return Tensor._op(BroadcastTo, a, op_args=(shape,), constant=constant)
