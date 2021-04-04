from numbers import Real
from typing import Optional

from numpy import ndarray

from mygrad.math.misc.funcs import maximum, minimum
from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike

__all__ = ["hard_tanh"]


def hard_tanh(
    x: ArrayLike,
    *,
    lower_bound: Real = -1,
    upper_bound: Real = 1,
    constant: Optional[bool] = None,
) -> Tensor:
    """Returns the hard hyperbolic tangent function.

    The hard_tanh function is `lower_bound` where `x` <= `lower_bound`, `upper_bound` where
    `x` >= `upper_bound`, and `x` where `lower_bound` < `x` < `upper_bound`.

    Parameters
    ----------
    x : ArrayLike
        The input, to which to apply the hard tanh function.

    lower_bound : Real, optional (default=-1)
        The lower bound on the hard tanh.

    upper_bound : Real, optional (default=1)
        The upper bound on the hard tanh.

    constant : Optional[bool]
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor
        The result of applying the "hard-tanh" function elementwise to `x`.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet.activations import hard_tanh
    >>> x = mg.arange(-5, 6)
    >>> x
    Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    >>> y = hard_tanh(x, lower_bound=-3, upper_bound=3); y
    Tensor([-3, -3, -3, -2, -1,  0,  1,  2,  3,  3,  3])
    >>> y.backward()
    >>> x.grad
    array([0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0.])

    .. plot::

       >>> import mygrad as mg
       >>> from mygrad.nnet.activations import hard_tanh
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-6, 6, 100)
       >>> y = hard_tanh(x, lower_bound=-3, upper_bound=3)
       >>> plt.title("hard_tanh(x, lower_bound=-3, upper_bound=3)")
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    if isinstance(lower_bound, (ndarray, Tensor)):
        lower_bound = lower_bound.item()

    if isinstance(upper_bound, (ndarray, Tensor)):
        upper_bound = upper_bound.item()

    if not isinstance(lower_bound, Real):
        raise TypeError(
            f"`lower_bound` must be a real-valued scalar, got {lower_bound} (type { type(lower_bound)})"
        )

    if not isinstance(upper_bound, Real):
        raise TypeError(
            f"`upper_bound` must be a real-valued scalar, got {upper_bound} (type {type(upper_bound)})"
        )

    return maximum(
        lower_bound, minimum(x, upper_bound, constant=constant), constant=constant
    )
