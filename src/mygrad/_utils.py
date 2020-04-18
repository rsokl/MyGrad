from numbers import Real
from typing import Any

import numpy as np

__all__ = ["is_invalid_gradient", "reduce_broadcast", "SkipGradient"]


class SkipGradient(Exception):
    """ The gradient for the current tensor-label pair has already
    been computed, scaled, and back-propped, skip gradient calculation."""


def reduce_broadcast(grad, var_shape):
    """ Sum-reduce axes of `grad` so its shape matches `var_shape.

        This the appropriate mechanism for backpropagating a gradient
        through an operation in which broadcasting occurred for the
        given variable.

        Parameters
        ----------
        grad : numpy.ndarray
        var_shape : Tuple[int, ...]

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            The dimensionality of the gradient cannot be less than
            that of its associated variable."""
    if grad.shape == var_shape:
        return grad

    if grad.ndim != len(var_shape):
        if grad.ndim < len(var_shape):
            raise ValueError(
                f"The dimensionality of the gradient of the broadcasted "
                f"operation ({grad.ndim}) is less than that of its associated variable "
                f"({len(var_shape)})"
            )
        grad = grad.sum(axis=tuple(range(grad.ndim - len(var_shape))))

    keepdims = tuple(n for n, i in enumerate(grad.shape) if i != var_shape[n])
    if keepdims:
        grad = grad.sum(axis=keepdims, keepdims=True)

    return grad


def is_invalid_gradient(grad: Any) -> bool:
    """Returns ``True`` if ``grad`` is not array-like.

    Parameters
    ----------
    grad : Any


    Returns
    -------
    ``True`` if ``grad`` is invalid"""
    return not isinstance(grad, (np.ndarray, Real)) or not np.issubdtype(
        np.asarray(grad).dtype, np.number
    )
