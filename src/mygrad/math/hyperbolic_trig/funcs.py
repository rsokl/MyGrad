from typing import Optional, Union

from numpy import ndarray

from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike, DTypeLikeReals, Mask
from mygrad.ufuncs import ufunc_creator

from .ops import *

__all__ = [
    "arccosh",
    "arccoth",
    "arccsch",
    "arcsinh",
    "arctanh",
    "cosh",
    "coth",
    "csch",
    "sech",
    "sinh",
    "tanh",
]


@ufunc_creator(Sinh)
def sinh(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Hyperbolic sine, element-wise.

    This docstring was adapted from that of numpy.sinh [1]_

    Parameters
    ----------
    x : ArrayLike
        Input tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.
        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    y : Tensor
        The corresponding hyperbolic sine values.


    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.sinh.html

    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972, pg. 83.

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.sinh(0)
    Tensor(0.0)

    >>> # Example of providing the optional output tensor
    >>> out1 = mg.tensor([0], dtype='d')
    >>> out2 = mg.sinh([0.1], out=out1)
    >>> out2
    Tensor([0.10016675])
    >>> out2 is out1
    True

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-4, 4, 100)
       >>> y = mg.arcsinh(x)
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


@ufunc_creator(Cosh)
def cosh(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Hyperbolic cosine, element-wise.

    This docstring was adapted from that of numpy.cosh [1]_

    Parameters
    ----------
    x : ArrayLike
        Input tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    out : Tensor
        Output tensor of same shape as `x`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.cosh.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.linspace(-2, 2, 10)
    >>> mg.cosh(x); y
    Tensor([3.76219569, 2.47439497, 1.68346238, 1.23057558, 1.02479314,
        1.02479314, 1.23057558, 1.68346238, 2.47439497, 3.76219569])

    >>> y.backward()  # compute d(cosh)/dx
    >>> x.grad
    array([-3.62686041, -2.26332289, -1.35426939, -0.71715846, -0.22405573,
        0.22405573,  0.71715846,  1.35426939,  2.26332289,  3.62686041])

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-4, 4, 100)
       >>> y = mg.cosh(x)
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


@ufunc_creator(Tanh)
def tanh(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Hyperbolic tangent, element-wise.

    This docstring was adapted from that of numpy.tanh [3]_

    Parameters
    ----------
    x : ArrayLike
        Input tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    y : Tensor
        The corresponding hyperbolic tangent values.


    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
           New York, NY: Dover, 1972, pg. 83.
           http://www.math.sfu.ca/~cbm/aands/

    .. [2] Wikipedia, "Hyperbolic function",
           https://en.wikipedia.org/wiki/Hyperbolic_function

    .. [3] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.tanh.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.linspace(-2, 2, 10)
    >>> y = mg.tanh(x); y
    Tensor([-0.96402758, -0.9146975 , -0.8044548 , -0.58278295, -0.21863508,
         0.21863508,  0.58278295,  0.8044548 ,  0.9146975 ,  0.96402758])

    >>> y.backward()  # compute d(tanh)/dx
    >>> x.grad
    array([0.07065082, 0.16332849, 0.35285247, 0.66036404, 0.9521987 ,
           0.9521987 , 0.66036404, 0.35285247, 0.16332849, 0.07065082])

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-4, 4, 100)
       >>> y = mg.tanh(x)
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


@ufunc_creator(Arcsinh)
def arcsinh(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Inverse hyperbolic sine, element-wise.

    This docstring was adapted from that of numpy.arcsinh [3]_

    Parameters
    ----------
    x : ArrayLike
        Input tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.
        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.

        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    out : Tensor
        Tensor of the same shape as `x`.

    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    returns ``nan`` and sets the `invalid` floating point error flag.

    The inverse hyperbolic sine is also known as `asinh` or ``sinh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arcsinh
    .. [3] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.arcsinh([mg.e, 10.0])
    Tensor([ 1.72538256,  2.99822295])

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-10, 10, 100)
       >>> y = mg.arcsinh(x)
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


@ufunc_creator(Arccosh)
def arccosh(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Inverse hyperbolic cosine, element-wise.

    This docstring was adapted from that of numpy.arccosh [3]_

    Parameters
    ----------
    x : ArrayLike
        Input tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.
        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    arccosh : Tensor
        Tensor of the same shape as `x`.



    See Also
    --------
    cosh, arcsinh, sinh, arctanh, tanh

    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]` and the real part in
    ``[0, inf]``.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arccosh
    .. [3] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.arccosh([mg.e, 10.0])
    Tensor([ 1.65745445,  2.99322285])
    >>> mg.arccosh(1)
    Tensor(0.0)

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(1.1, 10, 100)
       >>> y = mg.arccosh(x)
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


@ufunc_creator(Arctanh)
def arctanh(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Inverse hyperbolic tangent, element-wise.

    This docstring was adapted from that of numpy.arctanh [3]_

    Parameters
    ----------
    x : ArrayLike
        Input tensor.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).
        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.
        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    out : Tensor
        Tensor of the same shape as `x`.

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    The inverse hyperbolic tangent is also known as `atanh` or ``tanh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arctanh
    .. [3] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.arctanh([0, -0.5])
    Tensor([ 0.        , -0.54930614])

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-.9, .9, 100)
       >>> y = mg.arctanh(x)
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


def arccoth(a, *, constant=None):
    """``f(a) -> arccoth(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccoth, a, constant=constant)


def arccsch(a, *, constant=None):
    """``f(a) -> arccsch(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccsch, a, constant=constant)


def coth(a, *, constant=None):
    """``f(a) -> coth(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Coth, a, constant=constant)


def csch(a, *, constant=None):
    """``f(a) -> csch(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Csch, a, constant=constant)


def sech(a, *, constant=None):
    """``f(a) -> sech(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Sech, a, constant=constant)
