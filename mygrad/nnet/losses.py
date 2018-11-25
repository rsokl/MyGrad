from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
import numpy as np
from scipy.misc import logsumexp
from numbers import Real

__all__ = ["multiclass_hinge", "softmax_crossentropy", "margin_ranking_loss"]


class MulticlassHinge(Operation):
    def __call__(self, x, y_true, hinge=1.):
        """ Computes the average multiclass hinge loss

        Parameters
        ----------
        x : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        y_true : numpy.ndarray, shape=(N,)
            The correct class-index, in [0, C), for each datum.

        Returns
        -------
        loss : mygrad.Tensor
            The average multiclass hinge loss"""
        if x.ndim != 2:
            raise ValueError("`x` must be a 2D tensor, got {}D".format(x.ndim))
        if y_true.ndim != 1:
            raise ValueError("`y_true` must be a 1D tensor of integers, got {}D".format(y_true.ndim))

        self.variables = (x,)
        if isinstance(y_true, Tensor):
            y_true = y_true.data

        scores = x.data
        correct_labels = (range(len(y_true)), y_true)
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
    """ Computes the average multiclass hinge loss.

    Parameters
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
    loss : mygrad.Tensor
        The average (over N) multiclass hinge loss

    Raises
    ------
    ValueError
        Bad dimensionalities for ``x`` or ``y_true``
    """
    return Tensor._op(MulticlassHinge, x, op_args=(y_true, hinge), constant=constant)


class SoftmaxCrossEntropy(Operation):
    """ Given the classification scores of C classes for N pieces of data,
        computes the NxC softmax classification probabilities. The
        cross entropy is then computed by using the true classification labels.
        
        log-softmax is used for improved numerical stability"""
    def __call__(self, x, y_true):
        """
        Parameters
        ----------
        x : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        y_true : Sequence[int]
            The correct class-indices, in [0, C), for each datum.

        Returns
        -------
        loss : mygrad.Tensor
            The average softmax loss

        Raises
        ------
        ValueError
            Bad dimensionalities for ``x`` or ``y_true``
        """
        if x.ndim != 2:
            raise ValueError("`x` must be a 2D tensor, got {}D".format(x.ndim))
        if y_true.ndim != 1:
            raise ValueError("`y_true` must be a 1D tensor of integers, got {}D".format(y_true.ndim))

        self.variables = (x,)
        if isinstance(y_true, Tensor):
            y_true = y_true.data

        scores = x.data
        log_softmax = scores - logsumexp(scores, axis=-1, keepdims=True)
        label_locs = (range(len(scores)), y_true)
        loss = -np.sum(log_softmax[label_locs]) / scores.shape[0]
        
        self.back = np.exp(log_softmax)
        self.back[label_locs] -= 1.
        self.back /= scores.shape[0]
        return loss

    def backward_var(self, grad, index, **kwargs):
        return grad * self.back


def softmax_crossentropy(x, y_true, constant=False):
    r""" Given the classification scores of C classes for N pieces of data,
    computes the NxC softmax classification probabilities. The
    cross entropy is then computed by using the true classification labels
    and is averaged over the N pieces of data.

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
    loss : mygrad.Tensor
        The average softmax loss

    Raises
    ------
    ValueError
        Bad dimensionalities for ``x`` or ``y_true``

    Notes
    -----
    - :math:`N` is the number of samples in the batch.
    - :math:`C` is the number of possible classes for which scores are provided.

    Given the shape-:math:`(N, C)` tensor of scores, ``x``, the softmax classification
    probabilities are computed. That is, the score for class-:math:`k` of a given datum
    (:math:`s_{k}`) is normalized using the 'softmax' transformation:

    .. math::
        p_{k} = \frac{e^{s_k}}{\sum_{i=1}^{C}{e^{s_i}}}

    This produces the "prediction probability distribution", :math:`p`, for each datum.
    The cross-entropy loss for that datum is then computed according to the true class-index
    for that datum, as reported in ``y_true``. That is the "true probability distribution",
    :math:`t`, for the datum is :math:`1` for the correct class-index and :math:`0` elsewhere.
    The cross-entropy loss for that datum is thus:

    .. math::
       l = - \sum_{k=1}^{C}{t_{k} \log{p_{k}}}

    Having computed each per-datum cross entropy loss, this function then returns the loss
    averaged over all :math:`N` pieces of data:

    .. math::
       L = \frac{1}{N}\sum_{i=1}^{N}{l_{i}}

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import softmax_crossentropy

    Let's take a simple case where N=1, and C=3. We'll thus make up classification
    scores for a single datum. Suppose the scores are identical for the three classes
    and that the true class is class-0:

    >>> x = mg.Tensor([[2., 2., 2.]])  # a shape-(1, 3) tensor of scores
    >>> y_true = mg.Tensor([0])  # the correct class for this datum is class-0

    Because the scores are identical for all three classes, the softmax normalization
    will simply produce :math:`p = [\frac{1}{3}, \frac{1}{3}, \frac{1}{3}]`. Because
    class-0 is the "true" class, :math:`t = [1., 0., 0.]`. Thus our softmax cross-entropy
    loss should be:

    .. math::
      -(1 \times \log{\frac{1}{3}} + 0 \times \log{\frac{1}{3}} + 0 \times \log{\frac{1}{3}})
      = \log(3) \approx 1.099

    Let's see that this is what ``softmax_crossentropy`` returns:

    >>> softmax_crossentropy(x, y_true)
    Tensor(1.09861229)

    Similarly, suppose a datum's scores are :math:`[0, 0, 10^6]`, then the softmax normalization
    will return :math:`p \approx [0., 0., 1.]`. If the true class for this datum is class-2, then
    the loss should be nearly 0, since :math:`p` and :math:`t` are essentially identical:

    .. math::
      -(0 \times \log{0} + 0 \times \log{0} + 1 \times \log{1})
      = -\log(1) = 0

    Now, let's construct ``x`` and ``y_true`` so that they incorporate the scores/labels for
    both of the data that we have considered:

    >>> x = mg.Tensor([[2., 2.,  2.],  # a shape-(2, 3) tensor of scores
    ...                [0., 0., 1E6]])
    >>> y_true = mg.Tensor([0, 2])     # the class IDs for the two data

    ``softmax_crossentropy(x, y_true)`` will return the average loss of these two data,
    :math:`\frac{1}{2}(1.099 + 0) \approx 0.55`:

    >>> softmax_crossentropy(x, y_true)
    Tensor(0.54930614)
    """
    return Tensor._op(SoftmaxCrossEntropy, x, op_args=(y_true,), constant=constant)


class MarginRanking(Operation):
    def __call__(self, x1, x2, y, margin):
        """
        Computes the margin ranking loss between ``x1``
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
    loss : mygrad.Tensor
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
