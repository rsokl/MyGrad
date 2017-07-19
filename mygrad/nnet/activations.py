import numpy as np
from scipy.misc import logsumexp

from ..operations.operation_base import Operation
from ..tensor_base import Tensor

__all__ = ['tanh', 'sigmoid', 'relu', 'softmax', 'logsoftmax']


class Tanh(Operation):
    def __call__(self, a):
        self.a = a
        self.tanh = np.tanh(a.data)
        return self.tanh

    def backward_a(self, grad):
        self.a.backward(grad * (1 - self.tanh ** 2))


def tanh(x):
    """ f(x) = tanh(x)

        Parameters
        ----------
        x : Union[Tensor, array_like]

        Returns
        -------
        Tensor """
    return Tensor._op(Tanh, x)


class Sigmoid(Operation):
    def __call__(self, a):
        self.a = a
        x = -1. * a.data
        np.exp(x, out=x)
        x += 1
        np.reciprocal(x, out=x)
        self.sigmoid = x
        return self.sigmoid

    def backward_a(self, grad):
        self.a.backward(grad * self.sigmoid * (1. - self.sigmoid))


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
        self.a = a
        self.back = np.asarray(a > 0, dtype=self.a.dtype)
        return a.data * self.back

    def backward_a(self, grad):
        return self.a.backward(grad * self.back)


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
        self.a = a
        x = a.data
        assert 0 < a.ndim < 3

        self.__kw = dict(axis=1, keepdims=True) if a.ndim == 2 else dict(axis=None, keepdims=False)

        x = x - x.max(**self.__kw)
        np.exp(x, out=x)
        x /= x.sum(**self.__kw)
        return x

    def backward_a(self, grad):
        soft = self(self.a)
        sg = soft * grad
        self.a.backward(sg - soft * np.sum(sg, **self.__kw))


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
        self.a = a
        x = a.data
        assert 0 < a.ndim < 3

        self.__kw = dict(axis=1, keepdims=True) if x.ndim == 2 else dict(axis=None, keepdims=False)
        return x - logsumexp(x, **self.__kw)

    def backward_a(self, grad):
        x = self.a.data

        soft = x - x.max(**self.__kw)
        np.exp(soft, out=soft)
        soft /= soft.sum(**self.__kw)

        self.a.backward(grad - soft * np.sum(grad, **self.__kw))


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
