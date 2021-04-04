from typing import Optional

import numpy as np

from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike


class ReLu(Operation):
    def __call__(self, a):
        self.variables = (a,)
        self.back = np.asarray(a > 0, dtype=a.dtype)
        return a.data * self.back

    def backward_var(self, grad, index, **kwargs):
        return grad * self.back


def relu(x: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """
    Applies the recitfied linear unit activation function::

        f(x) = {x, x > 0
                0, x <= 0 }

    Parameters
    ----------
    x : ArrayLike
        relu is applied element-wise on ``x``.

    constant : Optional[bool]
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import relu
    >>> x = mg.linspace(-5, 5, 5)
    >>> x
    Tensor([-5. , -2.5,  0. ,  2.5,  5. ])
    >>> relu(x)
    Tensor([-0. , -0. ,  0. ,  2.5,  5. ])
    >>> relu(x).backward()
    >>> x.grad  # d(relu(x))/dx
    array([0., 0., 0., 1., 1.])

    .. plot::

       >>> import mygrad as mg
       >>> from mygrad.nnet.activations import relu
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-2, 2, 100)
       >>> y = relu(x)
       >>> plt.title("relu(x)")
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    return Tensor._op(ReLu, x, constant=constant)
