from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
import numpy as np
from scipy.misc import logsumexp
from numbers import Real

__all__ = ["multiclass_hinge", "softmax_crossentropy", "margin_ranking_loss"]


class MulticlassHinge(Operation):
    def __call__(self, a, y, hinge=1.):
        """ Parameters
            ----------
            a : mygrad.Tensor, shape=(N, C)
                The C class scores for each of the N pieces of data.

            y : numpy.ndarray, shape=(N,)
                The correct class-index, in [0, C), for each datum.

            Returns
            -------
            The average multiclass hinge loss"""
        self.variables = (a,)
        scores = a.data
        correct_labels = (range(len(y)), y)
        correct_class_scores = scores[correct_labels]  # Nx1

        M = scores - correct_class_scores[:, np.newaxis] + hinge  # NxC margins
        not_thresh = np.where(M <= 0)
        Lij = M
        Lij[not_thresh] = 0
        Lij[correct_labels] = 0

        TMP = np.ones(M.shape, dtype=float)
        TMP[not_thresh] = 0
        TMP[correct_labels] = 0  # NxC; 1 where margin > 0
        TMP[correct_labels] = -1 * TMP.sum(axis=-1)
        self.back = TMP
        self.back /= scores.shape[0]
        return np.sum(Lij) / scores.shape[0]

    def backward_var(self, grad, index, **kwargs):
        return grad * self.back


def multiclass_hinge(x, y_true, hinge=1., constant=False):
    """ Parameters
        ----------
        x : array_like, shape=(N, K)
            The K class scores for each of the N pieces of data.

        y : array_like, shape=(N,)
            The correct class-indices, in [0, K), for each datum.

        hinge : float
            The size of the "hinge" outside of which a nonzero loss
            is incurred.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        The average multiclass hinge loss"""
    return Tensor._op(MulticlassHinge, x, op_args=(y_true, hinge), constant=constant)


class SoftmaxCrossEntropy(Operation):
    """ Given the classification scores of C classes for N pieces of data,
        computes the NxC softmax classification probabilities. The
        cross entropy is then computed by using the true classification labels.
        
        log-softmax is used for improved numerical stability"""
    def __call__(self, a, y):
        """ Parameters
            ----------
            a : mygrad.Tensor, shape=(N, C)
                The C class scores for each of the N pieces of data.

            y : Sequence[int]
                The correct class-indices, in [0, C), for each datum.
                
            Returns
            -------
            The average softmax loss"""
        self.variables = (a,)
        scores = np.copy(a.data)
        log_softmax = scores - logsumexp(scores, axis=-1, keepdims=True)
        label_locs = (range(len(scores)), y)
        loss = -np.sum(log_softmax[label_locs]) / scores.shape[0]
        
        self.back = np.exp(log_softmax)
        self.back[label_locs] -= 1.
        self.back /= scores.shape[0]
        return loss

    def backward_var(self, grad, index, **kwargs):
        return grad * self.back


def softmax_crossentropy(x, y_true, constant=False):
    """ Given the classification scores of C classes for N pieces of data,
        computes the NxC softmax classification probabilities. The
        cross entropy is then computed by using the true classification labels.
        
        log-softmax is used for improved numerical stability.
        
        Parameters
        ----------
        x : array_like, shape=(N, C)
            The C class scores for each of the N pieces of data.

        y_true : array_like, shape=(N,)
            The correct class-indices, in [0, C), for each datum.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        The average softmax loss"""
    return Tensor._op(SoftmaxCrossEntropy, x, op_args=(y_true,), constant=constant)


class MarginRanking(Operation):
    def __call__(self, x1, x2, y, margin):
        """

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
        loss[not_thresh] = 0.

        self._grad = np.ones_like(M)
        self._grad[not_thresh] = 0.
        self._grad /= M.size
        return np.mean(loss)

    def backward_var(self, grad, index, **kwargs):
        sign = -self.y if index == 0 else self.y
        return grad * (sign * self._grad)


def margin_ranking_loss(x1, x2, y, margin, constant=False):
    """
    Computes the margin average margin ranking loss.

    Equivalent to:
             mg.mean(mg.maximum(0, margin - y * (x1 - x2)))

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
    assert 0 < x1.ndim < 3, "`x1` must have shape (N,) or (N, D)"
    assert x1.shape == x2.shape, "`x1` and `x2` must have the same shape"
    assert np.issubdtype(x1.dtype, np.floating), "`x1` must contain floats"
    assert isinstance(margin, Real) and margin >= 0, "`margin` must be a non-negative scalar"
    if isinstance(y, Tensor):
        y = y.data

    y = np.asarray(y)
    assert y.ndim == 0 or (y.ndim == 1 and len(y) == len(x1)), "`y` must be a scalar or shape-(N,) array of ones"
    if y.ndim:
        assert y.size == 1 or len(y) == len(x1)
        if x1.ndim == 2:
            y = y.reshape(-1, 1)
    return Tensor._op(MarginRanking, x1, x2, op_args=(y, margin), constant=constant)
