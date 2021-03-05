from typing import Optional, Tuple, Union

from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike

from .ops import MoveAxis, Roll, SwapAxes, Transpose

__all__ = ["transpose", "moveaxis", "swapaxes", "roll"]


@implements_numpy_override
def transpose(a: ArrayLike, *axes: int, constant: Optional[bool] = None) -> Tensor:
    """Permute the dimensions of a tensor.

    Parameters
    ----------
    a : ArrayLike
        The tensor to be transposed

    axes : int
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

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
        `a` with its axes permuted.  A new tensor is returned.

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.tensor([[1, 2], [3, 4]])
    >>> a
    Tensor([[1, 2],
            [3, 4]])
    >>> a.transpose()
    Tensor([[1, 3],
            [2, 4]])
    >>> a.transpose((1, 0))
    Tensor([[1, 3],
            [2, 4]])
    >>> a.transpose(1, 0)
    Tensor([[1, 3],
            [2, 4]])"""
    if not axes:
        axes = None
    elif hasattr(axes[0], "__iter__") or axes[0] is None:
        if len(axes) > 1:
            raise TypeError(
                f"'{type(axes[0])}' object cannot be interpreted as an integer"
            )
        axes = axes[0]
    return Tensor._op(Transpose, a, op_args=(axes,), constant=constant)


@implements_numpy_override
def moveaxis(
    a: ArrayLike,
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]],
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """Move axes of a tensor to new positions. Other axes remain in their
    original order.


    Parameters
    ----------
    a : ArrayLike
        The array whose axes should be reordered.

    source : Union[int, Sequence[int]]
        Original positions of the axes to move. These must be unique.

    destination : Union[int, Sequence[int]]
        Destination positions for each of the original axes. These must also be
        unique.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.
    Returns
    -------
    result : mygrad.Tensor
        Array with moved axes. This array is a view of the input array..

    Examples
    --------
    >>> from mygrad import zeros, moveaxis
    >>> x = zeros((3, 4, 5))
    >>> moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> moveaxis(x, -1, 0).shape
    (5, 3, 4)
    >>> moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)"""
    return Tensor._op(MoveAxis, a, op_args=(source, destination), constant=constant)


@implements_numpy_override
def swapaxes(
    a: ArrayLike, axis1: int, axis2: int, *, constant: Optional[bool] = None
) -> Tensor:
    """Interchange two axes of a tensor.

    Parameters
    ----------
    a : ArrayLike
        Input array.

    axis1 : int
        First axis.

    axis2 : int
        Second axis.

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

    Examples
    --------
    >>> from mygrad import Tensor, swapaxes
    >>> x = Tensor([[1, 2, 3]])
    >>> swapaxes(x, 0, 1)
    Tensor([[1],
           [2],
           [3]])
    >>> x = Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    >>> x
    Tensor([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> swapaxes(x, 0, 2)
    Tensor([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])
    """
    return Tensor._op(SwapAxes, a, op_args=(axis1, axis2), constant=constant)


@implements_numpy_override
def roll(
    a: ArrayLike,
    shift: Union[int, Tuple[int, ...]],
    axis=None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    Roll tensor elements along a given axis.

    Elements that roll beyond the end of an axis "wrap back around" to the beginning.

    This docstring was adapted from ``numpy.roll``

    Parameters
    ----------
    a : ArrayLike
        Input tensor.

    shift : Union[int, Tuple[int, ...]]
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.

    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

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
        Output array, with the same shape as `a`.

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.arange(10)
    >>> mg.roll(x, 2)
    Tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> x2 = mg.reshape(x, (2,5))
    >>> x2
    Tensor([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> mg.roll(x2, 1)
    Tensor([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> mg.roll(x2, 1, axis=0)
    Tensor([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> mg.roll(x2, 1, axis=1)
    Tensor([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    """
    return Tensor._op(
        Roll, a, op_kwargs=dict(shift=shift, axis=axis), constant=constant
    )
