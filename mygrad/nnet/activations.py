import numpy as np
from scipy.misc import logsumexp

from mygrad.operations.multivar_operations import Operation
from mygrad.tensor_base import Tensor
from mygrad import tanh

__all__ = ['tanh', 'sigmoid', 'relu', 'softmax', 'logsoftmax']


class Sigmoid(Operation):
    def __call__(self, a):
        self.variables = (a,)
        x = -1. * a.data
        np.exp(x, out=x)
        x += 1
        np.reciprocal(x, out=x)
        self.sigmoid = x
        return self.sigmoid

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.sigmoid * (1. - self.sigmoid), **kwargs)


def sigmoid(x):
    """ f(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        x : Union[Tensor, array_like]

        Returns
        -------
        Tensor """
    return Tensor._op(Sigmoid, x)


class ReLu(Operation):
    def __call__(self, a):
        self.variables = (a,)
        self.back = np.asarray(a > 0, dtype=a.dtype)
        return a.data * self.back

    def backward_var(self, grad, index, **kwargs):
        self.variables[index].backward(grad * self.back, **kwargs)


def relu(x):
    """ f(x) = {x, x > 0
                0, x <= 0 }

        Parameters
        ----------
        x : Union[Tensor, array_like]

        Returns
        -------
        Tensor """
    return Tensor._op(ReLu, x)


class Softmax(Operation):
    scalar_only = True

    def __call__(self, a):
        self.variables = (a,)
        x = a.data
        assert 0 < a.ndim < 3

        self.__kw = dict(axis=1, keepdims=True) if a.ndim == 2 else dict(axis=None, keepdims=False)

        x = x - x.max(**self.__kw)
        np.exp(x, out=x)
        x /= x.sum(**self.__kw)
        return x

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        soft = self(a)
        sg = soft * grad
        a.backward(sg - soft * np.sum(sg, **self.__kw), **kwargs)


def softmax(x):
    """ f(x) = exp(x) / sum( exp(x) )

        Compute the softmax over a 1D tensor of data, or over the respective rows
        of a 2D tensor

        Parameters
        ----------
        x : Union[Tensor, array_like]

        Returns
        -------
        Tensor """
    return Tensor._op(Softmax, x)


class LogSoftmax(Operation):
    scalar_only = True

    def __call__(self, a):
        self.variables = (a,)
        x = a.data
        assert 0 < a.ndim < 3

        self.__kw = dict(axis=1, keepdims=True) if x.ndim == 2 else dict(axis=None, keepdims=False)
        return x - logsumexp(x, **self.__kw)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        x = a.data

        soft = x - x.max(**self.__kw)
        np.exp(soft, out=soft)
        soft /= soft.sum(**self.__kw)

        a.backward(grad - soft * np.sum(grad, **self.__kw), **kwargs)


def logsoftmax(x):
    """ f(x) = log ( exp(x) / sum( exp(x) ) )

        Compute the log-softmax over a 1D tensor of data, or over the respective rows
        of a 2D tensor

        Parameters
        ----------
        x : Union[Tensor, array_like]

        Returns
        -------
        Tensor """
    return Tensor._op(LogSoftmax, x)
