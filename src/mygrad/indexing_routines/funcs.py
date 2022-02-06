from typing import Optional

import numpy as np

from mygrad.operation_base import _NoValue
from mygrad.tensor_base import Tensor, asarray, implements_numpy_override
from mygrad.typing import ArrayLike

from .ops import Where

__all__ = ["where"]


@implements_numpy_override()
def where(
    condition: ArrayLike,
    x: ArrayLike = _NoValue,
    y: ArrayLike = _NoValue,
    *,
    constant: Optional[bool] = None
) -> Tensor:
    """
    where(condition, [x, y])

    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only ``condition`` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. The rest of this
        documentation covers only the case where all three arguments are
        provided.

    This docstring was adapted from that of ``numpy.where``.

    Parameters
    ----------
    condition : ArrayLike, bool
        Where True, yield `x`, otherwise yield ``y``. ``x``, ``y``
        and `condition` need to be broadcastable to some shape.

    x : ArrayLike
        Values from which to chosen where ``condition`` is ``True``.

    y : ArrayLike
       Values from which to chosen where ``condition`` is ``False``.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    out : mygrad.Tensor
        A tensor with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.arange(10)
    >>> a
    Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mg.where(a < 5, a, 10*a)
    Tensor([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    This can be used on multidimensional tensors too:

    >>> mg.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    Tensor([[1, 8],
            [3, 4]])

    The shapes of x, y, and the condition are broadcast together:

    >>> x, y = np.ogrid[:3, :4]
    >>> mg.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
    Tensor([[10,  0,  0,  0],
            [10, 11,  1,  1],
            [10, 11, 12,  2]])

    >>> a = mg.Tensor([[0, 1, 2],
    ...                [0, 2, 4],
    ...                [0, 3, 6]])
    >>> mg.where(a < 4, a, -1)  # -1 is broadcast
    Tensor([[ 0,  1,  2],
            [ 0,  2, -1],
            [ 0,  3, -1]])
    """
    if x is _NoValue and y is _NoValue:
        return np.where(asarray(condition))

    if x is _NoValue or y is _NoValue:
        raise ValueError("either both or neither of x and y should be given")

    return Tensor._op(
        Where, x, y, op_kwargs={"condition": condition}, constant=constant
    )
