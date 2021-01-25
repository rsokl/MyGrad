import numpy as np

from mygrad import Tensor
from mygrad.operation_base import Operation

__all__ = ["selu"]


_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946


class SELU(Operation):
    """Returns the scaled exponential linear activation (SELU) elementwise along x. The SELU is
    given by  λɑ(exp(x) - 1) for x < 0 and λx for x ≥ 0.

    Notes
    -----
    The SELU activation was proposed in the paper
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
    at https://arxiv.org/abs/1706.02515
    """

    def __call__(self, x):
        """
        Parameters
        ----------
        x : mygrad.Tensor
            Input data.

        Returns
        -------
        numpy.ndarray
            The SELU function applied to `x` elementwise.
        """
        self.variables = (x,)

        x = x.data
        self.exp = _ALPHA * (np.exp(x) - 1)
        return _SCALE * np.where(x < 0, self.exp, x)

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index]
        return grad * _SCALE * np.where(x.data < 0, self.exp + _ALPHA, 1)


def selu(x, constant=False):
    """Returns the scaled exponential linear activation (SELU) elementwise along x.

    The SELU is given by  λɑ(exp(x) - 1) for x < 0 and λx for x ≥ 0.

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        The SELU function applied to `x` elementwise.

    Notes
    -----
    The SELU activation was proposed in the paper
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
    at https://arxiv.org/abs/1706.02515

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet.activations import selu
    >>> x = mg.arange(-5, 6)
    >>> x
    Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    >>> y = elu(x, alpha=0.1); y
    Tensor([-1.74625336, -1.72589863, -1.67056873, -1.52016647, -1.11133074,
         0.        ,  1.05070099,  2.10140197,  3.15210296,  4.20280395,
         5.25350494])
    """
    return Tensor._op(SELU, x, constant=constant)
