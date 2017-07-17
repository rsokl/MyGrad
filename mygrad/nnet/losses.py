from ..operations.operation_base import Operation
from ..tensor_base import Tensor
import numpy as np

__all__ = ["multiclass_hinge"]


class MulticlassHinge(Operation):
    def __call__(self, a, y, hinge=1.):
        """ Parameters
            ----------
            a : pygrad.Tensor, shape=(N, K)
                The K class scores for each of the N pieces of data.

            y : numpy.ndarray, shape=(N,)
                The correct class-index, in [0, K), for each datum.

            hinge : float, optional (default=1.0)
                The hinge-margin for the loss.

            Returns
            -------
            The average multiclass hinge loss"""
        # STUDENT CODE HERE
        pass

    def backward_a(self, grad):
        # STUDENT CODE HERE
        pass


def multiclass_hinge(x, y_true, hinge=1.):
    """ Parameters
        ----------
        x : mygrad.Tensor, shape=(N, K)
            The K class scores for each of the N pieces of data.

        y : Sequence[int]
            The correct class-indices, in [0, K), for each datum.
        Returns
        -------
        The average multiclass hinge loss"""
    return Tensor._op(MulticlassHinge, x, op_args=(y_true, hinge))
