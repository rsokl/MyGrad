from typing import Optional, Tuple, Union

import numpy as np

from mygrad import Tensor
from mygrad.operation_base import Operation
from mygrad.typing import ArrayLike

__all__ = ["batchnorm"]


# TODO: Remove affine parameters from Operation
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

    def __call__(self, x, gamma, beta, *, eps):
        """
        y(x) = (x - E[x]) / sqrt(Var[x} + eps)
        batchnorm(x) = gamma * y(x) + beta

        Parameters
        ----------
        x : mygrad.Tensor
        gamma : Optional[mygrad.Tensor]
        beta : Optional[mygrad.Tensor]
        eps : Real
           A small non-negative number.

        Returns
        -------
        numpy.ndarray
        """
        normed_dims = tuple(i for i in range(x.ndim) if i != 1)
        keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))

        self.variables = tuple(i for i in (x, gamma, beta))

        if gamma.size == 0:
            gamma = None
        if beta.size == 0:
            beta = None

        self.gamma = gamma
        self.beta = beta

        x = x.data
        self.x_norm = None  # required for backprop through gamma
        self.mean = x.mean(axis=normed_dims)
        self.var = x.var(axis=normed_dims)

        if eps:
            self.var += eps

        y = x - self.mean.reshape(keepdims_shape)
        self._std = np.sqrt(self.var).reshape(keepdims_shape)  # sqrt(var + eps)
        y /= self._std
        self.x_norm = y
        # optional affine transformation
        if gamma is not None:
            gamma = gamma.data
            # must copy `y` to prevent mutation of `self.x_norm`
            y = y * gamma.reshape(keepdims_shape)

        if beta is not None:
            beta = beta.data
            y = y + beta.reshape(keepdims_shape)
        return y

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[0].data
        if index == 0:  # backprop through x
            normed_dims = tuple(i for i in range(x.ndim) if i != 1)
            keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))
            N = x.size / x.shape[1]

            # all sums carried over non-channel dims
            # (1/sqrt(var + eps)) * [dL - dL.mean() - (1/N)*x_norm*(x_norm @ dL)]
            grad_ = grad - np.mean(grad, axis=normed_dims, keepdims=True)

            rterm = self.x_norm * np.reshape(
                np.einsum(grad, range(x.ndim), self.x_norm, range(x.ndim), [1]),
                keepdims_shape,
            )
            rterm /= N
            grad_ -= rterm
            grad_ /= self._std
            if (
                self.gamma is not None
            ):  # backprop through optional affine transformation
                gamma = self.gamma.data
                grad_ *= gamma.reshape(keepdims_shape)
            return grad_

        elif index == 1 and self.gamma is not None:  # backprop through gamma
            return np.einsum(grad, range(x.ndim), self.x_norm, range(x.ndim), [1])

        elif (index == 1 and self.gamma is None) or index == 2:
            normed_dims = tuple(i for i in range(x.ndim) if i != 1)
            return grad.sum(axis=normed_dims)
        else:  # pragma: no cover
            raise IndexError


def batchnorm(
    x: ArrayLike,
    *,
    gamma: Optional[ArrayLike] = None,
    beta: Optional[ArrayLike] = None,
    eps: float,
    constant: Optional[bool] = None
) -> Tensor:
    """
    Performs batch normalization on ``x``::

                 y(x) = (x - E[x]) / sqrt(Var[x] + eps)
                 batchnorm(x) = gamma * y(x) + beta

    Where :math:`E[x]` and :math:`Var[x]` represent the mean and variance, respectively,
    over axis-1 of ``x``. The subsequent affine transformation on ``y``
    is optional.

    Parameters
    ----------
    x : array_like, shape=(N, C, ...)
        The batch to be normalized within each entry of C

    gamma : Optional[array_like], shape=(C,)
        Optional per-channel scaling factors to be applied after the
        normalization step.

    beta  : Optional[array_like], shape=(C,)
        Optional per-channel scaling bias factors to be applied after the
        normalization step.

    eps : Real
       A small non-negative number.

    constant : bool, optional (default=False)
        If True, the resulting Tensor is a constant.

    Returns
    -------
    mygrad.Tensor
        The batch-normalized data.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import batchnorm
    >>> x = mg.Tensor([1., 4., 1.]).reshape(3, 1)
    >>> batchnorm(x, eps=0)
    Tensor([[-0.70710678],
            [ 1.41421356],
            [-0.70710678]])
    """
    # pass gamma and beta as empty arrays if they are not supplied
    if gamma is None:
        gamma = np.array([])
    if beta is None:
        beta = np.array([])
    return Tensor._op(
        BatchNorm, x, gamma, beta, op_kwargs=dict(eps=eps), constant=constant
    )
