import numpy as np

from mygrad import Tensor, asarray

from .ops import Where

__all__ = ["where"]


class _UniqueIdentifier:
    def __init__(self, identifier):
        self.identifier = identifier

    def __repr__(self):  # pragma: nocover
        return self.identifier


not_set = _UniqueIdentifier("not_set")


def where(condition, x=not_set, y=not_set, *, constant=None):
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
    condition : array_like, bool
        Where True, yield `x`, otherwise yield ``y``. ``x``, ``y``
        and `condition` need to be broadcastable to some shape.

    x : array_like
        Values from which to chosen where ``condition`` is ``True``.

    y : array_like
       Values from which to chosen where ``condition`` is ``False``.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

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
    if x is not_set and y is not_set:
        return np.where(asarray(condition))

    if x is not_set or y is not_set:
        raise ValueError("either both or neither of x and y should be given")

    return Tensor._op(
        Where, x, y, op_kwargs=dict(condition=condition), constant=constant
    )
