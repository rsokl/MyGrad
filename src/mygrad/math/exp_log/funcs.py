from typing import Optional, Union

from numpy import ndarray

from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike, DTypeLikeReals, Mask
from mygrad.ufuncs import ufunc_creator

from .ops import Exp, Exp2, Expm1, Log, Log1p, Log2, Log10, Logaddexp, Logaddexp2

__all__ = [
    "exp",
    "exp2",
    "expm1",
    "logaddexp",
    "logaddexp2",
    "log",
    "log2",
    "log10",
    "log1p",
]


@ufunc_creator(Exp)
def exp(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Calculate the exponential of all elements in the input tensor.

    This docstring was adapted from that of numpy.exp [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input values.

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
    exp : Tensor
        ``f(x1)`` computed element-wise

    See Also
    --------
    expm1 : Calculate ``exp(x) - 1`` for all elements in the tensor.
    exp2  : Calculate ``2**x`` for all elements in the tensor.

    Notes
    -----
    The irrational number ``e`` is also known as Euler's number.  It is
    approximately 2.718281, and is the base of the natural logarithm,
    ``ln`` (this means that, if :math:`x = \ln y = \log_e y`,
    then :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.exp.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.tensor(1.)
    >>> f = mg.exp(x); f  # f(1.)
    Tensor(2.71828183)

    Evaluate df/dx at ``x = 1``.

    >>> f.backward()
    >>> x.grad
    >>> x.grad  # df/dx @ x=1
    array(2.71828183)
    """
    ...


@ufunc_creator(Exp2)
def exp2(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Calculate `2**p` for all `p` in the input tensor.

    This docstring was adapted from that of numpy.exp2 [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input values.

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
    exp2 : Tensor
        ``f(x1)`` computed element-wise

    See Also
    --------
    power


    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.exp2.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.exp2([2., 3.])
    Tensor([ 4.,  8.])
    """
    ...


@ufunc_creator(Expm1)
def expm1(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Calculate ``exp(x) - 1`` for all elements in the tensor.

    This docstring was adapted from that of numpy.expm1 [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input values.

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
    expm1 : Tensor
        ``f(x1)`` computed element-wise

    See Also
    --------
    log1p : ``log(1 + x)``, the inverse of expm1.

    Notes
    -----
    This function provides greater precision than ``exp(x) - 1``
    for small values of ``x``.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.expm1.html

    Examples
    --------
    The true value of ``exp(1e-10) - 1`` is ``1.00000000005e-10`` to
    about 32 significant digits. This example shows the superiority of
    expm1 in this case.

    >>> import mygrad as mg
    >>> mg.expm1(1e-10)
    Tensor(1.00000000005e-10)
    >>> mg.exp(1e-10) - 1
    Tensor(1.000000082740371e-10)
    """
    ...


@ufunc_creator(Log)
def log(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Natural logarithm, element-wise.

    The natural logarithm ``log`` is the inverse of the exponential function,
    so that ``log(exp(x)) = x``. The natural logarithm is logarithm in base
    ``e``.

    This docstring was adapted from that of numpy.log [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input value.

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
    log : Tensor
        ``f(x1)`` computed element-wise

    See Also
    --------
    log10, log2, log1p

    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log` always returns real output. For
    each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.log.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.log([1, mg.e, mg.e**2, 0])
    array([  0.,   1.,   2., -Inf])
    """
    ...


@ufunc_creator(Log2)
def log2(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Base-2 logarithm applied elementwise to the tensor.

    This docstring was adapted from that of numpy.log2 [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input values.

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
    log2 : Tensor
        ``f(x1)`` computed element-wise

    See Also
    --------
    log, log10, log1p

    Notes
    -----
    For real-valued input data types, `log2` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.log2.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.tensor([0, 1, 2, 2**4])
    >>> mg.log2(x)
    Tensor([-Inf,   0.,   1.,   4.])
    """
    ...


@ufunc_creator(Log10)
def log10(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return the base 10 logarithm of the input tensor, element-wise.

    This docstring was adapted from that of numpy.log10 [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input values.

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
    log10 : Tensor
        ``f(x1)`` computed element-wise

    Notes
    -----
    For real-valued input data types, `log10` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.log10.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.log10([1e-15, -3.])
    Tensor([-15.,  nan])
    """
    ...


@ufunc_creator(Log1p)
def log1p(
    x1: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Return the natural logarithm of one plus the input tensor, element-wise.

    Calculates ``log(1 + x)``.

    This docstring was adapted from that of numpy.log1p [1]_

    Parameters
    ----------
    x1 : ArrayLike
        Input values.

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
    log1p : Tensor
        ``f(x1)`` computed element-wise

    See Also
    --------
    expm1 : ``exp(x) - 1``, the inverse of `log1p`.

    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.

    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.log1p.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.log1p(1e-99)
    1e-99
    >>> mg.log(1 + 1e-99)
    0.0
    """
    ...


@ufunc_creator(Logaddexp)
def logaddexp(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Logarithm of the sum of exponentiations of the inputs.

    Calculates ``log(exp(x1) + exp(x2))``. This function is useful in
    statistics where the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers.  In such cases
    the logarithm of the calculated probability is stored. This function
    allows adding probabilities stored in such a fashion.

    This docstring was adapted from that of numpy.logaddexp [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        Input values.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).

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
    logaddexp : Tensor
        Logarithm of ``exp(x1) + exp(x2)``.

    See Also
    --------
    logaddexp2: Logarithm of the sum of exponentiations of inputs in base 2.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html

    Examples
    --------
    >>> import mygrad as mg
    >>> prob1 = mg.log(1e-50)
    >>> prob2 = mg.log(2.5e-50)
    >>> prob12 = mg.logaddexp(prob1, prob2)
    >>> prob12
    Tensor(-113.87649168120691)
    >>> mg.exp(prob12)
    Tensor(3.5000000000000057e-50)
    """
    ...


@ufunc_creator(Logaddexp2)
def logaddexp2(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:
    """Logarithm of the sum of exponentiations of the inputs in base-2.

    Calculates ``log2(2**x1 + 2**x2)``. This function is useful in machine
    learning when the calculated probabilities of events may be so small as
    to exceed the range of normal floating point numbers.  In such cases
    the base-2 logarithm of the calculated probability can be used instead.
    This function allows adding probabilities stored in such a fashion.

    This docstring was adapted from that of numpy.logaddexp2 [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        Input values.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).

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
    logaddexp2 : Tensor
        Base-2 logarithm of ``2**x1 + 2**x2``.

    See Also
    --------
    logaddexp: Logarithm of the sum of exponentiations of the inputs.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html

    Examples
    --------
    >>> import mygrad as mg
    >>> prob1 = mg.log2(1e-50)
    >>> prob2 = mg.log2(2.5e-50)
    >>> prob12 = mg.logaddexp2(prob1, prob2)
    >>> prob1, prob2, prob12
    (Tensor(-166.09640474436813), Tensor(-164.77447664948076), Tensor(-164.28904982231052))
    >>> 2 ** prob12
    Tensor(3.4999999999999914e-50)
    """
    ...
