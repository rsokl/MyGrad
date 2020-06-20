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
    """
    return Tensor._op(ELU, x, op_args=(alpha,), constant=constant)
