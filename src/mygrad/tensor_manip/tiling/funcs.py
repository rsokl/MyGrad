from typing import Optional, Sequence, Union

from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike

from .ops import Repeat

__all__ = ["repeat"]


@implements_numpy_override
def repeat(
    a: ArrayLike,
    repeats: Union[int, Sequence[int]],
    axis: Optional[int] = None,
    *,
    constant: Optional[bool] = None,
) -> Tensor:
    """
    Repeat elements of a tensor.

    This docstring was adapted from ``numpy.repeat``

    Parameters
    ----------
    a : ArrayLike
        Input tensor.

    repeats : Union[int, Sequence[int]]
        The number of repetitions for each element. ``repeats``
        is broadcasted to fit the shape of the given axis.

    axis : Optional[int]
        The axis along which to repeat values. By default, use the
        flattened input array, and return a flat output tensor.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.
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
    return Tensor._op(Repeat, a, op_args=(repeats, axis), constant=constant)
