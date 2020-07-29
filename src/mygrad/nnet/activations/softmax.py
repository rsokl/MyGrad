import numpy as np

from mygrad.math._special import logsumexp as _logsumexp
from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor


def _softmax(x, kwargs):
    x = x - x.max(**kwargs)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float)
    if x.ndim > 0:
        np.exp(x, out=x)
        x /= x.sum(**kwargs)
    else:
        x = np.exp(x)
        x = x / x.sum(**kwargs)
    return x


class Softmax(Operation):
    scalar_only = True

    def __call__(self, a, axis=-1):
        self.variables = (a,)
        x = a.data

        self.__kw = dict(axis=axis, keepdims=True)

        return _softmax(x, self.__kw)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        soft = _softmax(a.data, self.__kw)
        sg = soft * grad
        return sg - soft * np.sum(sg, **self.__kw)


def softmax(x, axis=-1, constant=False):
    r"""
    Applies the softmax activation function::

        f(x) = exp(x) / sum( exp(x) )

    Compute the softmax over a 1D tensor of data, or along the
    respective rows of a 2D tensor

    Parameters
    ----------
    x : array_like, shape=(D,) or shape=(N,D)
        softmax is computed along the rows of ``x`` if
        ``x`` is a 2D array. Otherwise softmax is computed
        on the 1D ``x``.

    axis : Union[None, int, Tuple[int, ...]], optional (default=-1)
        The axis/axes over which to compute the softmax.
        By default, the softmax is computed over the trailing axis.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    - :math:`N` is the number of samples in the batch.
    - :math:`C` is the number of possible classes for which scores are provided.

    This implements a numerically-stable version of softmax, however
    log-softmax is still the more numerically stable activation function.

    Given the shape-:math:`(N, C)` tensor of scores, ``x``, the softmax classification
    probabilities are computed. That is, the score for class-:math:`k` of a given datum
    (:math:`s_{k}`) is normalized using the 'softmax' transformation:

    .. math::
        p_{k} = \frac{e^{s_k}}{\sum_{i=1}^{C}{e^{s_i}}}

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import softmax
    >>> x = mg.Tensor([[ 2.,  2.,  2.],
    ...                [2E50, 2E50,  1E50]])
    >>> softmax(x)
    Tensor([[0.33333333, 0.33333333, 0.33333333],
            [0.5       , 0.5       , 0.        ]])
    """
    return Tensor._op(Softmax, x, op_kwargs=dict(axis=axis), constant=constant)


class LogSoftmax(Softmax):
    scalar_only = True

    def __call__(self, a, axis=-1):
        self.variables = (a,)
        x = a.data

        self.__kw = dict(axis=axis, keepdims=True)
        return x - _logsumexp(x, **self.__kw)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        x = a.data
        soft = _softmax(x, self.__kw)
        return grad - soft * np.sum(grad, **self.__kw)


def logsoftmax(x, axis=-1, constant=False):
    r"""
    Applies the log-softmax activation function::

        f(x) = log ( exp(x) / sum( exp(x) ) )

    Compute the softmax over a 1D tensor of data, or along the
    respective rows of a 2D tensor

    Parameters
    ----------
    x : array_like, shape=(D,) or shape=(N,D)
        log-softmax is computed along the rows of ``x`` if
        ``x`` is a 2D array. Otherwise log-softmax is computed
        on the 1D ``x``.

    axis : Union[None, int, Tuple[int, ...]], optional (default=-1)
        The axis/axes over which to compute the log-softmax.
        By default, the log-softmax is computed over the trailing axis.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Notes
    -----
    - :math:`N` is the number of samples in the batch.
    - :math:`C` is the number of possible classes for which scores are provided.

    This implements a numerically-stable version of log-softmax, compared
    to the naive implementation using ``mygrad.log``, ``mygrad.exp``, and
    ``mygrad.sum``.

    Given the shape-:math:`(N, C)` tensor of scores, ``x``, the softmax classification
    probabilities are computed. That is, the score for class-:math:`k` of a given datum
    (:math:`s_{k}`) is normalized using the 'softmax' transformation:

    .. math::
        p_{k} = \log{\frac{e^{s_k}}{\sum_{i=1}^{C}{e^{s_i}}}}

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet import logsoftmax
    >>> x = mg.Tensor([[  2.,   2.,    2.],
    ...                [2E50, 2E50,  1E50]])
    >>> logsoftmax(x)
    Tensor([[-1.09861229e+00, -1.09861229e+00, -1.09861229e+00],
            [ 0.00000000e+00,  0.00000000e+00, -1.00000000e+50]])
    """
    return Tensor._op(LogSoftmax, x, op_kwargs=dict(axis=axis), constant=constant)
