from mygrad.operations.operation_base import Operation, BroadcastableOp
import numpy as np

__all__ = ["Exp", "Expm1",
           "Log", "Log2", "Log10", "Log1p",
           "Logaddexp", "Logaddexp2"]


class Exp(Operation):
    def __call__(self, a):
        """ f(a) -> exp(a)

            Parameters
            ----------
            a : mygrad.Tensor

            Returns
            -------
            numpy.ndarray"""
        self.variables = (a,)
        return np.exp(a.data)

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        var.backward(grad * np.exp(var.data))


class Expm1(Operation):
    """ f(a) -> exp(a) - 1

        This function provides greater precision than exp(x) - 1
        for small values of x."""

    def __call__(self, a):
        self.variables = (a,)
        return np.expm1(a.data)

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        var.backward(grad * np.exp(var.data))


class Logaddexp(BroadcastableOp):
    """f(a,b) -> log(exp(a) + exp(b))"""
    def __call__(self, a, b):
        self.variables = (a, b)
        out = np.logaddexp(a.data, b.data)
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            dLda = grad / (1 + np.exp(b.data - a.data))
            a.backward(dLda, **kwargs)
        else:
            dLdb = grad / (1 + np.exp(a.data - b.data))
            b.backward(dLdb, **kwargs)


class Logaddexp2(BroadcastableOp):
    """f(a,b) -> log2(exp(a) + exp(b))"""
    def __call__(self, a, b):
        self.variables = (a, b)
        out = np.logaddexp2(a.data, b.data)
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            dLda = grad / (1 + np.exp(b.data - a.data))
            dLda /= np.log(2)
            a.backward(dLda, **kwargs)
        else:
            dLdb = grad / (1 + np.exp(a.data - b.data))
            dLdb /= np.log(2)
            b.backward(dLdb, **kwargs)


class Log(Operation):
    """ f(a) -> ln(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.log(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / a.data, **kwargs)


class Log2(Operation):
    def __call__(self, a):
        """ f(a) -> log2(a)"""
        self.variables = (a,)
        return np.log2(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (a.data * np.log(2)), **kwargs)


class Log10(Operation):
    """ f(a) -> log10(a)"""
    def __call__(self, a):
        self.variables = (a,)
        return np.log10(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (a.data * np.log(10)), **kwargs)


class Log1p(Operation):
    """ f(a) -> ln(1 + a)

        log1p is accurate for x so small that 1 + x == 1 in
        floating-point accuracy."""
    def __call__(self, a):
        self.variables = (a,)
        return np.log1p(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad / (1 + a.data), **kwargs)
