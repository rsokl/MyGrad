from numbers import Real
from typing import Optional

from numpy import ndarray

from mygrad import maximum, minimum
from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike

__all__ = ["leaky_relu"]


def leaky_relu(
    x: ArrayLike, slope: float, *, constant: Optional[bool] = None
) -> Tensor:
    """Returns the leaky rectified linear activation elementwise along x.

    The leaky ReLU is given by `max(x, 0) + slope*min(x, 0)`.

    Parameters
    ----------
    x : ArrayLike
        Input data.

    slope : Union[Real, mygrad.Tensor]
        The slope of the negative activation.

    constant : Optional[bool]
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor
        The result of apply the "leaky relu" function elementwise to `x`.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet.activations import leaky_relu
    >>> x = mg.arange(-5, 6)
    >>> x
    Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    >>> y = leaky_relu(x, slope=0.1); y
    >>> Tensor([-0.5, -0.4, -0.3, -0.2, -0.1,  0. ,  1. ,  2. ,  3. ,  4. ,  5. ])
    >>> y.backward()
    >>> x.grad
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0. , 1. , 1. , 1. , 1. , 1. ])

    .. plot::

       >>> import mygrad as mg
       >>> from mygrad.nnet.activations import leaky_relu
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-2, 2, 100)
       >>> y = leaky_relu(x, slope=0.1)
       >>> plt.title("leaky_relu(x, slope=0.1)")
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    if isinstance(slope, (ndarray, Tensor)):
        slope = slope.item()

    if not isinstance(slope, Real):
        raise TypeError(
            f"`slope` must be a real-valued scalar, got {slope} (type { type(slope)})"
        )

    return maximum(x, 0, constant=constant) + slope * minimum(x, 0, constant=constant)
