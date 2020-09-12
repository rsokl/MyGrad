from numbers import Real

import numpy as np

from mygrad import Tensor
from mygrad.operation_base import Operation

__all__ = ["elu"]


class ELU(Operation):
    """ Returns the exponential linear activation (ELU) elementwise along x.

    The ELU is given by `ɑ(exp(x) - 1) for x < 0 and x for x ≥ 0`.
    """

    def __call__(self, x, alpha):
        """
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        alpha : Real
            The multiplicative factor on the negative activation.

        Returns
        -------
        numpy.ndarray
            The ELU function applied to `x` elementwise.
        """
        self.variables = (x,)

        x = x.data
        self.exp = alpha * (np.exp(x) - 1)
        self.alpha = alpha
        return np.where(x < 0, self.exp, x)

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        return grad * np.where(x.data < 0, self.exp + self.alpha, 1)


def elu(x, alpha, constant=False):
    """ Returns the exponential linear activation (ELU) elementwise along x.

    The ELU is given by  `ɑ(exp(x) - 1) for x < 0 and x for x ≥ 0`.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    alpha : Real
        The multiplicative factor on the negative activation.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        The ELU function applied to `x` elementwise.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet.activations import elu
    >>> x = mg.arange(-5, 6)
    >>> x
    Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    >>> y = elu(x, alpha=0.1); y
    Tensor([-0.09932621, -0.09816844, -0.09502129, -0.08646647, -0.06321206,
             0.        ,  1.        ,  2.        ,  3.        ,  4.        ,
             5.        ])
    >>> y.backward()
    >>> x.grad
    array([6.73794700e-04, 1.83156389e-03, 4.97870684e-03, 1.35335283e-02,
           3.67879441e-02, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
           1.00000000e+00, 1.00000000e+00, 1.00000000e+00])
    """
    if isinstance(alpha, (np.ndarray, Tensor)):
        alpha = alpha.item()

    if not isinstance(alpha, Real):
        raise TypeError(
            f"`alpha` must be a real-valued scalar, got {alpha} (type {type(alpha)})"
        )

    return Tensor._op(ELU, x, op_args=(alpha,), constant=constant)
