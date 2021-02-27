from typing import Optional

import numpy as np

from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike


class Sigmoid(Operation):
    def __call__(self, a):
        self.variables = (a,)
        x = np.asarray(-1.0 * a.data)
        np.exp(x, out=x)
        x += 1
        np.reciprocal(x, out=x)
        self.sigmoid = x
        return self.sigmoid

    def backward_var(self, grad, index, **kwargs):
        return grad * self.sigmoid * (1.0 - self.sigmoid)


def sigmoid(x: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """Applies the sigmoid activation function::

      f(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    x : array_like
        sigmoid is applied element-wise on ``x``.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import sigmoid
    >>> x = mg.linspace(-5, 5, 10)
    >>> sigmoid(x)
    Tensor([0.00669285, 0.02005754, 0.0585369 , 0.1588691 , 0.36457644,
        0.63542356, 0.8411309 , 0.9414631 , 0.97994246, 0.99330715])"""
    return Tensor._op(Sigmoid, x, constant=constant)
