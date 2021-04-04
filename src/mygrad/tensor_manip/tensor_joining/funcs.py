from typing import Optional, Sequence, Union

from numpy import ndarray

from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike, DTypeLikeReals

from .ops import Concatenate, Stack

__all__ = ["concatenate", "stack"]


@implements_numpy_override()
def concatenate(
    tensors: Sequence[ArrayLike],
    axis: Optional[int] = 0,
    out: Optional[Union[ndarray, Tensor]] = None,
    dtype: Optional[DTypeLikeReals] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    concatenate((t1, t2, ...), axis=0, out=None, *, constant=None)

    Join a sequence of tensors along an existing axis.

    This docstring was adapted from that of numpy.concatenate [1]_

    Parameters
    ----------
    tensors : Sequence[ArrayLike]
        The tensors must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).

    axis : Optional[int]
        The axis along which the tensors will be joined.  If axis is ``None``,
        tensors are flattened before use.  Default is 0.

    out : Optional[Union[ndarray, Tensor]]
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        If provided, the destination array will have this dtype. Cannot be provided
        together with ``out``.

        Requires numpy 1.20 or higher.

    Returns
    -------
    res : Tensor
        The concatenated tensor.

    See Also
    --------
    stack : Stack a sequence of tensors along a new axis.
    hstack : Stack tensors in sequence horizontally (column wise).
    vstack : Stack tensors in sequence vertically (row wise).
    dstack : Stack tensors in sequence depth wise (along third dimension).

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.tensor([[1, 2], [3, 4]])
    >>> b = mg.tensor([[5, 6]])
    >>> mg.concatenate((a, b), axis=0)
    Tensor([[1, 2],
           [3, 4],
           [5, 6]])
    >>> mg.concatenate((a, b.T), axis=1)
    Tensor([[1, 2, 5],
           [3, 4, 6]])
    >>> mg.concatenate((a, b), axis=None)
    Tensor([1, 2, 3, 4, 5, 6])
    """
    return Tensor._op(
        Concatenate,
        *tensors,
        op_kwargs={"axis": axis, "dtype": dtype},
        constant=constant,
        out=out,
    )


@implements_numpy_override()
def stack(
    tensors: Sequence[ArrayLike],
    axis: int = 0,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    stack((t1, t2, ...), axis=0, out=None, *, constant=None)

    Join a sequence of tensors along a new axis.

    This docstring was adapted from that of numpy.stack [1]_

    Parameters
    ----------
    tensors : Sequence[ArrayLike]
        Each tensor must have the same shape.

    axis : Optional[int]
        The axis in the result tensor along which the input tensors are stacked.

    out : Optional[Union[ndarray, Tensor]]
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    res : Tensor
        The stacked tensor has one more dimension than the input arrays.

    See Also
    --------
    concatenate : Join a sequence of tensors along an existing axis.
    hstack : Stack tensors in sequence horizontally (column wise).
    vstack : Stack tensors in sequence vertically (row wise).
    dstack : Stack tensors in sequence depth wise (along third dimension).

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.stack.html

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.tensor([1, 2, 3])
    >>> b = mg.tensor([-1, -2, -3])
    >>> mg.stack((a, b))
    Tensor([[ 1,  2,  3],
            [-1, -2, -3]])

    >>> mg.stack((a, b), axis=-1)
    Tensor([[1, -1],
            [2, -2],
            [3, -3]])
    """
    return Tensor._op(
        Stack,
        *tensors,
        op_kwargs={"axis": axis},
        constant=constant,
        out=out,
    )
