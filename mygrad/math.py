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
           "arcsech",
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
           "multiply",
           "multiply_sequence",
           "power",
           "sec",
           "sech",
           "sin",
           "sinh",
           "sqrt",
           "subtract",
           "tan",
           "tanh"]

def abs(a):
    """ f(a)-> |a|

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Abs, a)


def add(a, b):
    """ f(a, b) -> a + b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Add, a, b)


def add_sequence(*variables):
    """ f(a, b, ...) -> a + b + ...

        Parameters
        ----------
        variables : Sequence[Union[tensor-like, Number]]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(AddSequence, *variables)


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


def arcsech(a):
    """ f(a)-> arcsech(a)

        Parameters
        ----------
        a : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Arcsech, a)


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


def divide(a, b):
    """ f(a, b) -> a / b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Divide, a, b)


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


def multiply(a, b):
    """ f(a, b) -> a * b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Multiply, a, b)


def multiply_sequence(*variables):
    """ f(a, b, ...) -> a * b * ...

        Parameters
        ----------
        variables : Sequence[Union[tensor-like, Number]]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(MultiplySequence, *variables)


def power(a, b):
    """ f(a, b) -> a ** b

        Parameters
        ----------
        a : Union[tensor-like, Number]
        b : Union[tensor-like, Number]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Power, a, b)


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
