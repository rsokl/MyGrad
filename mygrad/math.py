from .tensor_base import Tensor
from .operations import *

__all__ = ["abs",
           "add",
           "add_sequence",
           "arccos",
           "arccosh",
           "arccot",
           "arccoth",
           "arccsc",
           "arccsch",
           "arcsec",
           "arcsin",
           "arcsinh",
           "arctan",
           "arctanh",
           "cbrt",
           "cos",
           "cosh",
           "cot",
           "coth",
           "csc",
           "csch",
           "divide",
           "exp",
           "log",
           "log2",
           "log10",
           "logaddexp",
           "max",
           "mean"
           "min"
           "multiply",
           "multiply_sequence",
           "power",
           "sec",
           "sech",
           "sin",
           "sinh",
           "sqrt",
           "subtract",
           "sum",
           "tan",
           "tanh"]


def sum(x, axis=None, keepdims=False):
    """ Sum of tensor elements over a given axis.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor]

        axis : Optional[int, Tuple[ints, ...]
            Axis or axes along which a sum is performed.  The default,
            axis=None, will sum all of the elements of the input tensor.  If
            axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, a sum is performed on all of the axes
            specified in the tuple instead of a single axis or all the axes as
            before.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input tensor.

        Returns
        -------
        sum_along_axis : Tensor
            A Tensor with the same shape as `self`, with the specified
            axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
            a 0-dim Tensor is returned."""
    return Tensor._op(Sum, x, op_args=(axis, keepdims))


def mean(x, axis=None, keepdims=False):
    """ Mean of tensor elements over a given axis.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor]

        axis : Optional[int, Tuple[ints, ...]
            Axis or axes along which a mean is performed.  The default,
            axis=None, will mean all of the elements of the input tensor.  If
            axis is negative it counts from the last to the first axis.

            If axis is a tuple of ints, a mean is performed on all of the axes
            specified in the tuple instead of a single axis or all the axes as
            before.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input tensor.

        Returns
        -------
        mean_along_axis : Tensor
            A Tensor with the same shape as `self`, with the specified
            axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
            a 0-dim Tensor is returned."""
    return Tensor._op(Mean, x, op_args=(axis, keepdims))


def max(x, axis=None, keepdims=False):
    """ Return the maximum of a tensor, or along its axes.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor]

        axis : Optional[int, Tuple[int, ...]]
            Axis or axes along which to operate. By default, flattened input is used.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `arr`.

        Returns
        -------
        max : Tensor
            Maximum of `a`. If `axis` is None, the result is a 0-D tensor."""
    return Tensor._op(MaxMin, x, op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin='max'))


def min(x, axis=None, keepdims=False):
    """ Return the minimum of a tensor, or along its axes.

        Parameters
        ----------
        axis : Optional[int, Tuple[int, ...]]
            Axis or axes along which to operate. By default, flattened input is used.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `arr`.

        Returns
        -------
        min : Tensor
            Minimum of `a`. If `axis` is None, the result is a 0-D tensor."""
    return Tensor._op(MaxMin, x, op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin='min'))


def abs(a):
    """ f(a)-> |a|

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Abs, a)





def arccos(a):
    """ f(a)-> arccos(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccos, a)


def arccosh(a):
    """ f(a)-> arccosh(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccosh, a)


def arccot(a):
    """ f(a)-> arccot(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccot, a)


def arccoth(a):
    """ f(a)-> arccoth(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccoth, a)


def arccsc(a):
    """ f(a)-> arccsc(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccsc, a)


def arccsch(a):
    """ f(a)-> arccsch(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arccsch, a)


def arcsec(a):
    """ f(a)-> arcsec(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arcsec, a)

def arcsin(a):
    """ f(a)-> arcsin(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arcsin, a)


def arcsinh(a):
    """ f(a)-> arcsinh(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arcsinh, a)


def arctan(a):
    """ f(a)-> arctan(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arctan, a)

def arctanh(a):
    """ f(a)-> arctanh(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arctanh, a)


def cbrt(a):
    """ f(a)-> cbrt(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cbrt, a)


def cos(a):
    """ f(a)-> cos(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cos, a)


def cosh(a):
    """ f(a)-> cosh(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cosh, a)


def cot(a):
    """ f(a)-> cot(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Cot, a)


def coth(a):
    """ f(a)-> coth(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Coth, a)


def csc(a):
    """ f(a)-> csc(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Csc, a)


def csch(a):
    """ f(a)-> csch(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Csch, a)


def exp(a):
    """ f(a)-> exp(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Exp, a)


def log(a):
    """ f(a)-> log(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Log, a)


def log2(a):
    """ f(a)-> log2(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Log2, a)


def log10(a):
    """ f(a)-> log10(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Log10, a)


def logaddexp(a, b):
    """ f(a, b)-> log(exp(a) + exp(b))

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Logaddexp, a, b)

def sec(a):
    """ f(a)-> sec(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sec, a)


def sech(a):
    """ f(a)-> sech(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sech, a)


def sin(a):
    """ f(a)-> sin(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sin, a)


def sinh(a):
    """ f(a)-> sinh(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sinh, a)


def sqrt(a):
    """ f(a)-> sqrt(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Sqrt, a)


def subtract(a, b):
    """ f(a, b) -> a - b
        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]
        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Subtract, a, b)


def tan(a):
    """ f(a)-> tan(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Tan, a)

def tanh(a):
    """ f(a)-> tanh(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Tanh, a)
