from numbers import Real

import numpy as np

import mygrad._utils.graph_tracking as _tracking
from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor, asarray


class MarginRanking(Operation):
    def __call__(self, x1, x2, y, margin):
        """Computes the margin ranking loss between ``x1``
        and ``x2``.

        Parameters
        ----------
        x1 : mygrad.Tensor, shape=(N,) or (N, D)

        x2 : mygrad.Tensor, shape=(N,) or (N, D)

        y : numpy.ndarray

        margin : float

        Returns
        -------
        numpy.ndarray, shape=()
        """
        self.variables = (x1, x2)
        x1 = x1.data
        x2 = x2.data

        self.y = y

        M = margin - self.y * (x1 - x2)
        not_thresh = M <= 0
        loss = M
        loss[not_thresh] = 0.0
        if _tracking.TRACK_GRAPH:
            self._grad = np.ones_like(M)
            self._grad[not_thresh] = 0.0
            self._grad /= M.size
        return np.mean(loss)

    def backward_var(self, grad, index, **kwargs):
        sign = -self.y if index == 0 else self.y
        return grad * (sign * self._grad)


def margin_ranking_loss(x1, x2, y, margin, *, constant=None):
    r"""Computes the margin average margin ranking loss.
    Equivalent to::

    >>> import mygrad as mg
    >>> mg.mean(mg.maximum(0, margin - y * (x1 - x2)))

    Parameters
    ----------
    x1 : array_like, shape=(N,) or (N, D)
        A batch of scores or descriptors to compare against those in `x2`

    x2 : array_like, shape=(N,) or (N, D)
        A batch of scores or descriptors to compare against those in `x1`

    y  : Union[int, array_like], scalar or shape=(N,)
        1 or -1. Specifies whether the margin is compared against `(x1 - x2)`
        or `(x2 - x1)`, for each of the N comparisons.

    margin : float
        A non-negative value to be used as the margin for the loss.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor, shape=()
        The mean margin ranking loss.
    """
    if not 0 < x1.ndim < 3:
        raise ValueError("`x1` must have shape (N,) or (N, D)")
    if not x1.shape == x2.shape:
        raise ValueError("`x1` and `x2` must have the same shape")
    if not np.issubdtype(x1.dtype, np.floating):
        raise TypeError("`x1` must contain floats")
    if not np.issubdtype(x2.dtype, np.floating):
        raise TypeError("`x2` must contain floats")
    if not isinstance(margin, Real) or margin < 0:
        raise ValueError("`margin` must be a non-negative scalar")

    y = asarray(y)

    if y.size == 1:
        y = np.array(y.item())

    if not y.ndim == 0 and not (y.ndim == 1 and len(y) == len(x1)):
        raise ValueError("`y` must be a scalar or shape-(N,) array of ones")

    if y.ndim:
        if x1.ndim == 2:
            y = y.reshape(-1, 1)
    return Tensor._op(MarginRanking, x1, x2, op_args=(y, margin), constant=constant)
