from typing import Optional, Sequence, Union

from mygrad.tensor_base import Tensor

from .ops import Repeat

__all__ = ["repeat"]


def repeat(
    a,
    repeats: Union[int, Sequence[int]],
    axis: Optional[int] = None,
    constant: bool = False,
) -> Tensor:
    """
    Repeat elements of a tensor.

    This docstring was adapted from ``numpy.repeat``

    Parameters
    ----------
    a : array_like
        Input tensor.

    repeats : Union[int, Sequence[int]]
        The number of repetitions for each element. ``repeats``
        is broadcasted to fit the shape of the given axis.

    axis : Optional[int]
        The axis along which to repeat values. By default, use the
        flattened input array, and return a flat output tensor.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it does not
        back-propagate a gradient).

    Returns
    -------
    repeated_tensor : Tensor
        Output tensor which has the same shape as `a`, except along
        the given axis.

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.repeat(3, 4)
    Tensor([3, 3, 3, 3])
    >>> x = mg.Tensor([[1, 2], [3, 4]])
    >>> mg.repeat(x, 2)
    Tensor([1, 1, 2, 2, 3, 3, 4, 4])
    >>> mg.repeat(x, 3, axis=1)
    Tensor([[1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4]])
    >>> mg.repeat(x, [1, 2], axis=0)
    Tensor([[1, 2],
            [3, 4],
            [3, 4]])
    """
    return Tensor._op(Repeat, a, op_args=(repeats, axis), force_constant=constant)
