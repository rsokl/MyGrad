import numpy as np

from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor

from ._utils import check_loss_inputs


class MulticlassHinge(Operation):
    def __call__(self, a, y, hinge=1.0):
        """ Computes the average multiclass hinge loss.

        Parameters
        ----------
        a : mygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.

        y : numpy.ndarray, shape=(N,)
            The correct class-index, in [0, C), for each datum.

        Returns
        -------
        The average multiclass hinge loss

        Raises
        ------
        TypeError
            `y_true` must be an integer-type array-like object

        ValueError
            `x` must be a 2-dimensional array-like object
            `y_true` must be a shape-(N,) array-like object"""

        check_loss_inputs(a, y)
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


def multiclass_hinge(x, y_true, hinge=1.0, constant=False):
    """ Computes the average multiclass hinge loss.

    Parameters
    ----------
    x : array_like, shape=(N, K)
        The K class scores for each of the N pieces of data.

    y_true : array_like, shape=(N,)
        The correct class-indices, in [0, K), for each datum.

    hinge : float
        The size of the "hinge" outside of which a nonzero loss
        is incurred.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    The average multiclass hinge loss

    Raises
    ------
    TypeError
        `y_true` must be an integer-type array-like object

    ValueError
        `x` must be a 2-dimensional array-like object
        `y_true` must be a shape-(N,) array-like object
    """
    return Tensor._op(MulticlassHinge, x, op_args=(y_true, hinge), constant=constant)
