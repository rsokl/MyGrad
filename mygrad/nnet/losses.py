from ..operations.operation_base import Operation
from ..tensor_base import Tensor
import numpy as np

__all__ = ["multiclass_hinge"]


class MulticlassHinge(Operation):
    def __call__(self, a, y, hinge=1.):
        """ Parameters
            ----------
            a : pygrad.Tensor, shape=(N, C)
                The C class scores for each of the N pieces of data.
            y : numpy.ndarray, shape=(N,)
                The correct class-index, in [0, C), for each datum.
            Returns
            -------
            The average multiclass hinge loss"""
        self.a = a
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

    def backward_a(self, grad):
        self.a.backward(grad * self.back)
        self.back = None


def multiclass_hinge(x, y_true, hinge=1.):
    """ Parameters
        ----------
        x : pygrad.Tensor, shape=(N, C)
            The C class scores for each of the N pieces of data.
        y : Sequence[int]
            The correct class-indices, in [0, C), for each datum.
        Returns
        -------
        The average multiclass hinge loss"""
    return Tensor._op(MulticlassHinge, x, y_true, hinge)
