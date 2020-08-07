from numbers import Real

import numpy as np

from mygrad import Tensor
from mygrad.nnet.activations import softmax
from mygrad.operation_base import Operation

from ._utils import check_loss_inputs

__all__ = ["softmax_focal_loss", "focal_loss"]


class FocalLoss(Operation):
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

    def __call__(self, class_probs, targets, alpha, gamma):
        """
        Parameters
        ----------
        class_probs : mygrad.Tensor, shape=(N, C)
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
        if isinstance(targets, Tensor):  # pragma: nocover
            targets = targets.data

        check_loss_inputs(class_probs, targets)

        self.variables = (class_probs,)
        if isinstance(class_probs, Tensor):
            class_probs = class_probs.data

        self.label_locs = (range(len(class_probs)), targets)

        pc = class_probs[self.label_locs]
        one_m_pc = np.clip(1 - pc, a_min=0, a_max=1)
        log_pc = np.log(pc)

        one_m_pc_gamma = one_m_pc ** gamma
        loss = -(alpha * one_m_pc_gamma * log_pc)

        self.back = np.zeros(class_probs.shape, dtype=np.float64)

        if np.isclose(gamma, 0, atol=1e-15):
            self.back[self.label_locs] -= alpha / pc
            return loss

        # dL/dp = -alpha * ( (1 - p)**g / p - g * (1 - p)**(g - 1) * log(p) )
        #
        # term 1: (1 - p)**g / p
        term1 = one_m_pc_gamma / pc  # (1 - p)**g / p

        # term 2: - g * (1 - p)**(g - 1) * log(p)
        if np.isclose(gamma, 1, rtol=1e-15):
            term2 = -log_pc
        elif gamma < 1:
            # For g < 1 and p -> 1, the 2nd term -> 0 via L'Hôpital's rule
            term2 = np.zeros(pc.shape, dtype=np.float64)
            pc_not_1 = ~np.isclose(one_m_pc, 0, atol=1e-15)
            term2[pc_not_1] = (
                -gamma * one_m_pc[pc_not_1] ** (gamma - 1) * log_pc[pc_not_1]
            )
        else:
            term2 = -gamma * one_m_pc ** (gamma - 1) * log_pc

        self.back[self.label_locs] -= alpha * (term1 + term2)
        return loss

    def backward_var(self, grad, index, **kwargs):
        self.back[self.label_locs] *= grad
        return self.back


def focal_loss(class_probs, targets, *, alpha=1, gamma=0, constant=False):
    r""" Return the per-datum focal loss.

    Parameters
    ----------
    class_probs : mygrad.Tensor, shape=(N, C)
        The C class probabilities for each of the N pieces of data.
        Each value is expected to lie on (0, 1]

    targets : Sequence[int], shape=(N,)
        The correct class indices, in [0, C), for each datum.

    alpha : Real, optional (default=1)
        The ɑ weighting factor in the loss formulation.

    gamma : Real, optional (default=0)
        The ɣ focusing parameter. Note that for Ɣ=0 and ɑ=1, this is cross-entropy loss.
        Must be a non-negative value.

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


    The focal loss for datum-:math:`i` is given by

    .. math::
        -\alpha \hat{y}_i(1-p_i)^\gamma\log(p_i)

    where :math:`\hat{y}_i` is one in correspondence to the label associated with the
    datum and 0 elsewhere. That is, if the label :math:`y_k` is 2 and
    there are four possible label values, then :math:`\hat{y}_k = (0, 0, 1, 0)`.

    It is recommended in the paper that you normalize by the number of foreground samples.
    """
    if not isinstance(gamma, Real) or gamma < 0:
        raise ValueError(f"`gamma` must be a non-negative number, got: {gamma}")

    return Tensor._op(
        FocalLoss, class_probs, op_args=(targets, alpha, gamma), constant=constant
    )


def softmax_focal_loss(scores, targets, *, alpha=1, gamma=0, constant=False):
    r"""
    Applies the softmax normalization to the input scores before computing the
    per-datum focal loss.

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
        Must be a non-negative value.

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

    The focal loss for datum-:math:`i` is given by

    .. math::
        -\alpha \hat{y}_i(1-p_i)^\gamma\log(p_i)

    where :math:`\hat{y}_i` is one in correspondence to the label associated with the
    datum and 0 elsewhere. That is, if the label :math:`y_k` is 2 and
    there are four possible label values, then :math:`\hat{y}_k = (0, 0, 1, 0)`.

    It is recommended in the paper that you normalize by the number of foreground samples.
    """
    return focal_loss(
        softmax(scores), targets=targets, alpha=alpha, gamma=gamma, constant=constant
    )
