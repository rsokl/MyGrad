from functools import reduce

import numpy as np

from mygrad.operation_base import BroadcastableOp, Operation

__all__ = [
    "Add",
    "_IAdd",
    "Subtract",
    "_ISubstract",
    "Multiply",
    "_IMultiply",
    "Divide",
    "_IDivide",
    "Reciprocal",
    "Power",
    "_IPower",
    "_IPow1",
    "Square",
    "_ISquare",
    "Positive",
    "Negative",
    "AddSequence",
    "MultiplySequence",
]


class Add(BroadcastableOp):
    def __call__(self, a, b):
        """ Performs 'add' forward-pass: f(a,b) -> a + b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """

        self.variables = (a, b)
        out = a.data + b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        return grad


class _IAdd(Add):
    def __call__(self, inplace_target, other) -> np.ndarray:
        """ Performs a += b

        Parameters
        ----------
        inplace_target : mygrad.Tensor
        other : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray

        Notes
        -----
        Note that inplace_target will be replaced with its un-mutated
        value by ``Tensor._inplace_op`` prior to backprop. Thus we need
        not worry about caching inplace_target and rewriting the backprop
        logic.

        However, also note that using this op outside of the context of
        `Tensor._inplace_op`` will lead to broken behavior"""

        self.variables = (inplace_target, other)
        inplace_target = inplace_target.data
        inplace_target += other.data
        return inplace_target


class Subtract(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a,b) -> a - b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            out : numpy.ndarray """
        self.variables = (a, b)
        out = a.data - b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        if index == 0:
            return grad
        else:
            return -grad


class _ISubstract(Subtract):
    def __call__(self, inplace_target, other) -> np.ndarray:
        """ Performs a -= b

        Parameters
        ----------
        inplace_target : mygrad.Tensor
        other : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray

        Notes
        -----
        Note that inplace_target will be replaced with its un-mutated
        value by ``Tensor._inplace_op`` prior to backprop. Thus we need
        not worry about caching inplace_target and rewriting the backprop
        logic.

        However, also note that using this op outside of the context of
        `Tensor._inplace_op`` will lead to broken behavior"""

        self.variables = (inplace_target, other)
        inplace_target = inplace_target.data
        inplace_target -= other.data
        return inplace_target


class Multiply(BroadcastableOp):
    def __call__(self, a, b):
        """ Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor"""
        self.variables = (a, b)
        out = a.data * b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            return grad * b.data
        elif index == 1:  # backprop through b
            return grad * a.data


class _IMultiply(Multiply):
    def __call__(self, inplace_target, other) -> np.ndarray:
        """ Performs a *= b

        Parameters
        ----------
        inplace_target : mygrad.Tensor
        other : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray

        Notes
        -----
        Note that inplace_target will be replaced with its un-mutated
        value by ``Tensor._inplace_op`` prior to backprop. Thus we need
        not worry about caching inplace_target and rewriting the backprop
        logic.

        However, also note that using this op outside of the context of
        `Tensor._inplace_op`` will lead to broken behavior"""

        self.variables = (inplace_target, other)
        inplace_target = inplace_target.data
        inplace_target *= other.data
        return inplace_target


class Divide(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a, b) -> a / b"""
        self.variables = (a, b)
        out = a.data / b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:  # backprop through a
            return grad / b.data
        else:  # broadcast through b
            return -grad * a.data / (b.data ** 2)


class _IDivide(Divide):
    def __call__(self, inplace_target, other) -> np.ndarray:
        """ Performs a /= b

        Parameters
        ----------
        inplace_target : mygrad.Tensor
        other : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray

        Notes
        -----
        Note that inplace_target will be replaced with its un-mutated
        value by ``Tensor._inplace_op`` prior to backprop. Thus we need
        not worry about caching inplace_target and rewriting the backprop
        logic.

        However, also note that using this op outside of the context of
        `Tensor._inplace_op`` will lead to broken behavior"""

        self.variables = (inplace_target, other)
        inplace_target = inplace_target.data
        inplace_target /= other.data
        return inplace_target


class Reciprocal(BroadcastableOp):
    def __call__(self, a):
        """ f(a) -> 1 / a"""
        self.variables = (a,)
        return np.reciprocal(a.data)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        return -grad * np.reciprocal(a.data ** 2)


class Power(BroadcastableOp):
    def __call__(self, a, b):
        """ f(a, b) -> a ** b

            Parameters
            ----------
            a: mygrad.Tensor
            b: mygrad.Tensor"""
        self.variables = (a, b)
        out = a.data ** b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        x, y = (i.data for i in self.variables)

        if index == 0:
            return grad * y * (x ** np.where(y, (y - 1), 1))
        else:
            return grad * (x ** y) * np.log(np.where(x, x, 1))


class _IPower(Power):
    def __call__(self, inplace_target, other) -> np.ndarray:
        """ Performs a **= b

        Parameters
        ----------
        inplace_target : mygrad.Tensor
        other : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray

        Notes
        -----
        Note that inplace_target will be replaced with its un-mutated
        value by ``Tensor._inplace_op`` prior to backprop. Thus we need
        not worry about caching inplace_target and rewriting the backprop
        logic.

        However, also note that using this op outside of the context of
        `Tensor._inplace_op`` will lead to broken behavior"""

        self.variables = (inplace_target, other)
        inplace_target = inplace_target.data
        inplace_target **= other.data
        return inplace_target


class Square(Operation):
    def __call__(self, a):
        """ f(a) -> a ** 2

            Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        return np.square(a.data)

    def backward_var(self, grad, index, **kwargs):
        grad = 2 * grad
        grad *= self.variables[index].data
        return grad


class _ISquare(Square):
    def __call__(self, inplace_target) -> np.ndarray:
        """ Performs a **= 2  (special case)

        Parameters
        ----------
        inplace_target : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray

        Notes
        -----
        Note that inplace_target will be replaced with its un-mutated
        value by ``Tensor._inplace_op`` prior to backprop. Thus we need
        not worry about caching inplace_target and rewriting the backprop
        logic.

        However, also note that using this op outside of the context of
        `Tensor._inplace_op`` will lead to broken behavior"""

        self.variables = (inplace_target,)
        inplace_target = inplace_target.data
        inplace_target **= 2
        return inplace_target


class _IPow1(Operation):
    def __call__(self, inplace_target) -> np.ndarray:
        """ Performs a **= 1  (special case)

        Parameters
        ----------
        inplace_target : mygrad.Tensor

        Returns
        -------
        inplace_target_data : numpy.ndarray
        """

        self.variables = (inplace_target,)
        return inplace_target.data

    def backward_var(self, grad: np.ndarray, index: int, **kwargs) -> np.ndarray:
        return grad


class Positive(Operation):
    """ f(a) = +a """

    def __call__(self, a, where=True):
        """
        Parameters
        ----------
        a: mygrad.Tensor

        where : array_like, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone."""
        self.variables = (a,)
        self.conf = dict(where=where)
        return np.positive(a.data, where=where)

    def backward_var(self, grad, index, **kwargs):
        return np.positive(grad, **self.conf)


class Negative(Operation):
    """ f(a) = -a """

    def __call__(self, a, where=True):
        """
        Parameters
        ----------
        a : mygrad.Tensor

        where : array_like, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone."""
        self.variables = (a,)
        self.conf = dict(where=where)
        return np.negative(a.data, where=where)

    def backward_var(self, grad, index, **kwargs):
        return np.negative(grad, **self.conf)


class AddSequence(BroadcastableOp):
    """Performs f(a, b, ..., z) = a + b + ... + z"""

    def __call__(self, *input_vars):
        assert len(input_vars) > 1, "`add_sequence` requires at least two operands"
        self.variables = input_vars
        out = sum(var.data for var in input_vars)
        return out

    def backward_var(self, grad, index, **kwargs):
        return grad


class MultiplySequence(BroadcastableOp):
    """ Performs f(a, b, ..., z) = a * b * ... * z"""

    def __call__(self, *input_vars):
        self.variables = input_vars
        assert 2 <= len(self.variables)

        out = reduce(lambda x, y: x * y, (var.data for var in input_vars))
        self._iszero = np.any(out == 0)
        return out

    def backward(self, grad, *, graph, **kwargs):
        """ Back-propagates the gradient through all of the operation's inputs. This needs to be updated
            by an operation if that operation takes more than 2 Tensor arguments."""
        if not self._iszero:
            self._product = grad * reduce(
                lambda x, y: x * y, (var.data for n, var in enumerate(self.variables))
            )
        else:
            self._product = None
        super().backward(grad, graph=graph, **kwargs)

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        if not self._iszero:
            return self._product / var.data
        else:
            return grad * reduce(
                lambda x, y: x * y,
                (var.data for n, var in enumerate(self.variables) if n != index),
            )
