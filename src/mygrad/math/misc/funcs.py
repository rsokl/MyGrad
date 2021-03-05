from typing import Optional, Union

from numpy import ndarray

from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike, DTypeLikeReals, Mask
from mygrad.ufuncs import ufunc_creator

from .ops import Abs, Cbrt, Maximum, Minimum, Sqrt

__all__ = ["abs", "absolute", "cbrt", "clip", "sqrt", "maximum", "minimum"]


@ufunc_creator(Abs)
def absolute(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """The absolute value, computed elementwise.

    This docstring was adapted from that of numpy.absolute [1]_

    Parameters
    ----------
    x : ArrayLike
        Input array.

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
    absolute : Tensor
        An ndarray containing the absolute value of
        each element in `x`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.absolute.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.array([-1.2, 1.2])
    >>> mg.absolute([-1.2, 1.2])
    Tensor([ 1.2,  1.2])

    Plot the function and its derivate over ``[-10, 10]``:

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-5, 5, 100)
       >>> y = mg.absolute(x)
       >>> plt.title("absolute(x)")
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


abs = absolute


@ufunc_creator(Sqrt)
def sqrt(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """The square root, elementwise.

    This docstring was adapted from that of numpy.sqrt [1]_

    Parameters
    ----------
    x : ArrayLike
        The values whose square-roots are required.

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
    y : ndarray
        A tensor of the same shape as `x`, containing the positive
        square-root of each element in `x`. Negative-valued inputs
        produce nans.


    Notes
    -----
    *sqrt* has--consistent with common convention--as its branch cut the
    real "interval" [`-inf`, 0), and is continuous from above on it.
    A branch cut is a curve in the complex plane across which a given
    complex function fails to be continuous.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.sqrt([1, 4, 9])
    Tensor([ 1.,  2.,  3.])

    >>> mg.sqrt([4, -1, mg.inf])
    Tensor([ 2., nan, inf])
    """
    ...


@ufunc_creator(Cbrt)
def cbrt(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """The cube root elementwise.

    This docstring was adapted from that of numpy.cbrt [1]_

    Parameters
    ----------
    x : ArrayLike
        The values whose cube-roots are computed.

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
    y : ndarray
        A tensor of the same shape as `x`, containing the cube
        cube-root of each element in `x`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.cbrt([1, 8, 27])
    Tensor([ 1.,  2.,  3.])
    """
    ...


@ufunc_creator(Maximum)
def maximum(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Pair-wise maximum of tensor elements.

    This docstring was adapted from that of numpy.maximum [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        The tensors holding the elements to be compared.
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
    y : Tensor
        The maximum of `x1` and `x2`, element-wise.

    See Also
    --------
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.

    Notes
    -----
    The maximum is equivalent to ``mg.where(x1 >= x2, x1, x2)`` when
    neither x1 nor x2 are nans, but it is faster and does proper
    broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.maximum.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.maximum([2, 3, 4], [1, 5, 2])
    Tensor([2, 5, 4])

    >>> mg.maximum(mg.eye(2), [0.5, 2]) # broadcasting
    Tensor([[ 1. ,  2. ],
           [ 0.5,  2. ]])

    >>> mg.maximum([mg.nan, 0, mg.nan], [0, mg.nan, mg.nan])
    Tensor([nan, nan, nan])
    >>> mg.maximum(mg.Inf, 1)
    Tensor(inf)
    """
    ...


@ufunc_creator(Minimum)
def minimum(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Pair-wise minimum of tensor elements.

    This docstring was adapted from that of numpy.minimum [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        The tensors holding the elements to be compared.
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
    y : Tensor
        The minimum of `x1` and `x2`, element-wise.

    See Also
    --------
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.

    Notes
    -----
    The minimum is equivalent to ``mg.where(x1 <= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster and does proper
    broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.minimum.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.minimum([2, 3, 4], [1, 5, 2])
    Tensor([1, 3, 2])

    >>> mg.minimum(mg.eye(2), [0.5, 2]) # broadcasting
    Tensor([[ 0.5,  0. ],
           [ 0. ,  1. ]])

    >>> mg.minimum([mg.nan, 0, mg.nan],[0, mg.nan, mg.nan])
    Tensor([nan, nan, nan])
    >>> mg.minimum(-mg.Inf, 1)
    Tensor(-inf)
    """
    ...


@implements_numpy_override
def clip(a, a_min, a_max, *, constant=None):
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Equivalent to `mg.minimum(a_max, mg.maximum(a, a_min))``.

    No check is performed to ensure ``a_min < a_max``.

    This docstring was adapted from that of `numpy.clip`

    Parameters
    ----------
    a : ArrayLike
        Array containing elements to clip.

    a_min : Optional[float, ArrayLike]
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.

    a_max : Optional[float, ArrayLike]
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`. If `a_min` or `a_max` are ArrayLike, then the three
        arrays will be broadcasted to match their shapes.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    Tensor
        A tensor with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.arange(10)
    >>> mg.clip(a, 1, 8)
    Tensor([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a
    Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mg.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    Tensor([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])"""
    if a_min is None and a_max is None:
        raise ValueError("`a_min` and `a_max` cannot both be set to `None`")

    if a_min is not None:
        a = maximum(a_min, a, constant=constant)

    if a_max is not None:
        a = minimum(a_max, a, constant=constant)

    return a
