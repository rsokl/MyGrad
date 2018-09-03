from mygrad.operation_base import Operation
from mygrad import Tensor
import numpy as np

__all__ = ["batchnorm"]


class BatchNorm(Operation):
    """
    Attributes
    ----------
    mean : numpy.ndarray
    var : numpy.ndarray

    Notes
    -----
    `mean` and `var` are bound as instance-attributes upon
    calling the batch-norm instance.
    """
    scalar_only = True

    def __call__(self, x, gamma, beta, eps):
        """
        y(x) = (x - E[x]) / sqrt(Var[x} + eps)
        batchnorm(x) = gamma * y(x) + beta

        Parameters
        ----------
        x : mygrad.Tensor
        gamma : mygrad.Tensor
        beta : mygrad.Tensor
        eps : Real
           A small non-negative number.

        Returns
        -------
        numpy.ndarray
        """
        self.variables = (x, gamma, beta)
        x, gamma, beta = x.data, gamma.data, beta.data
        self.mean = x.mean(axis=0)
        self.var = np.einsum("i...,i...->...", x, x)
        self.var /= x.shape[0]
        self.var -= self.mean ** 2
        if eps:
            self.var += eps

        y = x - np.expand_dims(self.mean, axis=0)
        y /= np.sqrt(self.var)
        self.x_norm = y
        y = y * gamma
        y += beta
        return y

    def backward_var(self, grad, index, **kwargs):

        if index == 0:  # backprop through x
            x = self.variables[0].data
            gamma = self.variables[1].data
            grad = grad - np.mean(grad, axis=0, keepdims=True)
            x_sub_Ex = x - self.mean
            rterm = x_sub_Ex / self.var
            rterm /= x.shape[0]
            rterm *= np.einsum("i...,i...->...", grad, x_sub_Ex)
            grad -= rterm
            grad /= np.sqrt(self.var)
            grad *= gamma
            return grad
        elif index == 1:  # backprop through gamma
            return np.einsum("i...,i...->...", grad, self.x_norm)
        elif index == 2:
            return grad.sum(axis=0)
        else:
            raise IndexError

def batchnorm(x, *, gamma, beta, eps, constant=False):
    return Tensor._op(BatchNorm, x, gamma, beta, op_kwargs=dict(eps=eps), constant=constant)


