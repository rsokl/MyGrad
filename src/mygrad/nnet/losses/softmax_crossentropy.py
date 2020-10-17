import numpy as np

import mygrad._utils.graph_tracking as _tracking
from mygrad.math._special import logsumexp
from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor

from ._utils import check_loss_inputs


class SoftmaxCrossEntropy(Operation):
    """ Given the classification scores of C classes for N pieces of data,
        computes the NxC softmax classification probabilities. The
        cross entropy is then computed by using the true classification labels.

        log-softmax is used for improved numerical stability"""

    def __call__(self, x, y_true):
        """ Parameters
            ----------
            x : mygrad.Tensor, shape=(N, C)
                The C class scores for each of the N pieces of data.

            y_true : Sequence[int]
                The correct class-indices, in [0, C), for each datum.

            Returns
            -------
            The average softmax loss"""
        if isinstance(y_true, Tensor):
            y_true = y_true.data

        check_loss_inputs(x, y_true)
        self.variables = (x,)
        scores = x.data
        log_softmax = scores - logsumexp(scores, axis=-1, keepdims=True)
        label_locs = (range(len(scores)), y_true)
        loss = -np.sum(log_softmax[label_locs]) / scores.shape[0]

        if _tracking.TRACK_GRAPH:
            self.back = np.exp(log_softmax)
            self.back[label_locs] -= 1.0
            self.back /= scores.shape[0]
        return loss

    def backward_var(self, grad, index, **kwargs):
        return grad * self.back


def softmax_crossentropy(x, y_true, constant=False):
    r""" Given the classification scores of C classes for N pieces of data,

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
