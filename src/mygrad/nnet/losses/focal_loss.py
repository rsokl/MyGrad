import numpy as np

from mygrad import Tensor, log
from mygrad.operation_base import Operation

from ._utils import check_loss_inputs

__all__ = ["softmax_focal_loss", "focal_loss"]


class SoftmaxFocalLoss(Operation):
    r""" Returns the per-datum focal loss as described in https://arxiv.org/abs/1708.02002
    which is given by -ɑ(1-p)ˠlog(p).

    Extended Description
    --------------------
    The focal loss is given by

    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}-\alpha \hat{y}_i(1-p_i)^\gamma\log(p_i)

    where :math:`N` is the number of elements in `x` and `y` and :math:`\hat{y}_i` is
    one where :math:`i` is the label of the element :math:`y_i` and 0 elsewhere. That is,
    if the label :math:`y_k` is 1 and there are four possible label values, then
    :math:`\hat{y}_k = (0, 1, 0, 0)`.

    It is recommended in the paper that you normalize by the number of foreground samples.
    """

    scalar_only = True

    def __call__(self, scores, targets, alpha, gamma):
        """
        Parameters
        ----------
        scores : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        targets : Union[mygrad.Tensor, Sequence[int]], shape=(N,)
            The correct class indices, in [0, C), for each datum.

        alpha : Real
            The ɑ weighting factor in the loss formulation.

        gamma : Real
            The ɣ focusing parameter.

        Returns
        -------
        numpy.ndarray
            The per-datum focal loss.
        """
        if isinstance(targets, Tensor):
            targets = targets.data

        check_loss_inputs(scores, targets)

        self.variables = (scores,)
        scores = np.copy(scores.data if isinstance(scores, Tensor) else scores)

        max_scores = np.max(scores, axis=1, keepdims=True)
        np.exp(scores - max_scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        label_locs = (range(len(scores)), targets)

        pc = scores[label_locs]
        one_m_pc = 1 - pc + 1e-14  # correct domain for when gamma < 1 and pc == 1
        log_pc = np.log(pc)

        loss = -(alpha * one_m_pc ** gamma * log_pc)

        self.back = scores
        self.back[label_locs] -= 1
        deriv = one_m_pc ** gamma - pc * gamma * one_m_pc ** (gamma - 1) * log_pc
        self.back *= deriv[:, np.newaxis]
        self.back *= alpha
        return loss

    def backward_var(self, grad, index, **kwargs):
        return grad[:, np.newaxis] * self.back


def softmax_focal_loss(scores, targets, *, alpha=1, gamma=0, constant=False):
    r"""
    Applies the softmax normalization to the input scores before computing the
    per-datum focal loss.

    Parameters
    ----------
    scores : mygrad.Tensor, shape=(N, C)
        The C class scores for each of the N pieces of data.

    targets : array_like, shape=(N,)
        The correct class indices, in [0, C), for each datum.

    alpha : Real, optional (default=1)
        The ɑ weighting factor in the loss formulation.

    gamma : Real, optional (default=0)
        The ɣ focusing parameter. Note that for Ɣ=0 and ɑ=1, this is cross-entropy loss.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor, shape=(N,)
        The per-datum focal loss.

    Notes
    -----
    The formulation for the focal loss introduced in https://arxiv.org/abs/1708.02002.
    It is given by -ɑ(1-p)ˠlog(p).


    The focal loss is given by

    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}-\alpha \hat{y}_i(1-p_i)^\gamma\log(p_i)

    where :math:`N` is the number of elements in `x` and `y` and :math:`\hat{y}_i` is
    one where :math:`i` is the label of the element :math:`y_i` and 0 elsewhere. That is,
    if the label :math:`y_k` is 1 and there are four possible label values, then
    :math:`\hat{y}_k = (0, 1, 0, 0)`.

    It is recommended in the paper that you normalize by the number of foreground samples.
    """
    return Tensor._op(
        SoftmaxFocalLoss, scores, op_args=(targets, alpha, gamma), constant=constant
    )


def focal_loss(scores, targets, *, alpha=1, gamma=0, constant=False):
    r""" Return the per-datum focal loss.

    Parameters
    ----------
    scores : mygrad.Tensor, shape=(N, C)
        The C class scores for each of the N pieces of data.

    targets : Sequence[int], shape=(N,)
        The correct class indices, in [0, C), for each datum.

    alpha : Real, optional (default=1)
        The ɑ weighting factor in the loss formulation.

    gamma : Real, optional (default=0)
        The ɣ focusing parameter. Note that for Ɣ=0 and ɑ=1, this is cross-entropy loss.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor, shape=(N,)
        The per-datum focal loss.

    Notes
    -----
    The formulation for the focal loss introduced in https://arxiv.org/abs/1708.02002.
    It is given by -ɑ(1-p)ˠlog(p).


    The focal loss is given by

    .. math::
        \frac{1}{N}\sum\limits_{1}^{N}-\alpha \hat{y}_i(1-p_i)^\gamma\log(p_i)

    where :math:`N` is the number of elements in `x` and `y` and :math:`\hat{y}_i` is
    one where :math:`i` is the label of the element :math:`y_i` and 0 elsewhere. That is,
    if the label :math:`y_k` is 1 and there are four possible label values, then
    :math:`\hat{y}_k = (0, 1, 0, 0)`.

    It is recommended in the paper that you normalize by the number of foreground samples.
    """
    if isinstance(targets, Tensor):
        targets = targets.data

    check_loss_inputs(scores, targets)

    label_locs = (range(len(targets)), targets)
    pc = scores[label_locs]
    return -(alpha * (1 - pc + 1e-14) ** gamma * log(pc, constant=constant))
