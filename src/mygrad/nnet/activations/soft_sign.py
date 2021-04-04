from typing import Optional

from mygrad import abs, divide
from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike

__all__ = ["soft_sign"]


def soft_sign(x: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """Returns the soft sign function x / (1 + |x|).

    Parameters
    ----------
    x : ArrayLike
        Input data.

    constant : boolean, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor
        The soft sign function applied to `x` elementwise.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet.activations import soft_sign
    >>> x = mg.arange(-5, 6)
    >>> x
    Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    >>> y = soft_sign(x); y
    Tensor([-0.83333333, -0.8       , -0.75      , -0.66666667, -0.5       ,
         0.        ,  0.5       ,  0.66666667,  0.75      ,  0.8       ,
         0.83333333])

    .. plot::

       >>> import mygrad as mg
       >>> from mygrad.nnet.activations import soft_sign
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-10, 10, 100)
       >>> y = soft_sign(x)
       >>> plt.title("soft_sign(x)")
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    return divide(x, 1 + abs(x), constant=constant)
