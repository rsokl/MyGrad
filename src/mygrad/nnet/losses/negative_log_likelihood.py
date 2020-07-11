import numpy as np

from mygrad import Tensor, mean

from ._utils import check_loss_inputs


def negative_log_likelihood(x, y_true, *, weights=None, constant=False):
    """ Returns the (weighted) negative log-likelihood loss between log-probabilities and y_true.

    Note that this does not compute a softmax, so you should input log-probabilities to this.
    See ``softmax_crossentropy`` if you need your loss to compute a softmax.

    Parameters
    ----------
    x : array_like, shape=(N, C)
        The C log-probabilities for each of the N pieces of data.

    y_true : array_like, shape=(N,)
        The correct class indices, in [0, C), for each datum.

    weights : array_like, shape=(C,) optional (default=None)
        The weighting factor to use on each class, or None.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor, shape=()
        The average (weighted) negative log-likelihood loss.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import negative_log_likelihood

    Let's take a simple case where N=1, and C=3. We'll thus make up classification
    scores for a single datum. Suppose the scores are identical for the three classes
    and that the true class is class-0, so that the log-probs are each 1/3:

    >>> logprob = mg.log(1 / 3).item()
    >>> x = mg.Tensor([[logprob, logprob, logprob]])  # a shape-(1, 3) tensor of log-probabilities
    >>> y_true = mg.Tensor([0])  # the correct class for this datum is class-0
    >>> negative_log_likelihood(x, y_true)
    Tensor(1.09861229)

    # log-probabilities where the prediction is highly-confident and correct
    >>> x = mg.Tensor([[0, -20, -20]])
    >>> negative_log_likelihood(x, y_true)
    Tensor(0.)

    # adding a class-weighting
    >>> x = mg.Tensor([[-4.6, -4.6, -0.02]])
    >>> weights = mg.Tensor([2, 1, 1])
    >>> negative_log_likelihood(x, y_true, weights=weights)
    Tensor(9.2)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    check_loss_inputs(x, y_true)

    if weights is None:
        weights = np.ones(x.shape[1])
    if isinstance(weights, Tensor):
        weights = weights.data

    if weights.ndim != 1 or weights.shape[0] != x.shape[1]:
        raise ValueError(
            "`weights` must be a shape-(C,) array: \n"
            f"\tExpected shape-{x.shape[1]}\n"
            f"\tGot shape-{y_true.shape}"
        )

    label_locs = (range(len(y_true)), y_true)
    factors = weights[y_true]
    return -mean(x[label_locs] * factors, constant=constant)
