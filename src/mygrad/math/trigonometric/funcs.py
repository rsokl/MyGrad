from typing import Optional, Union

from numpy import ndarray

from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike, DTypeLikeReals, Mask
from mygrad.ufuncs import ufunc_creator

from .ops import *

__all__ = [
    "sin",
    "sinc",
    "cos",
    "tan",
    "cot",
    "csc",
    "sec",
    "arccos",
    "arccsc",
    "arcsin",
    "arctan",
    "arcsec",
    "arccot",
    "arctan2",
]


@ufunc_creator(Sin)
def sin(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    r"""Trigonometric sine, element-wise.

    This docstring was adapted from that of numpy.sin [1]_

    Parameters
    ----------
    x : ArrayLike
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).

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
        The sine of each element of x.

    See Also
    --------
    arcsin, sinh, cos

    Notes
    -----
    The sine is one of the fundamental functions of trigonometry (the
    mathematical study of triangles).  Consider a circle of radius 1
    centered on the origin.  A ray comes in from the :math:`+x` axis, makes
    an angle at the origin (measured counter-clockwise from that axis), and
    departs from the origin.  The :math:`y` coordinate of the outgoing
    ray's intersection with the unit circle is the sine of that angle.  It
    ranges from -1 for :math:`x=3\pi / 2` to +1 for :math:`\pi / 2.`  The
    function has zeroes where the angle is a multiple of :math:`\pi`.
    Sines of angles between :math:`\pi` and :math:`2\pi` are negative.
    The numerous properties of the sine and related functions are included
    in any standard trigonometry text.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.sin.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.sin(mg.pi/2.)
    Tensor(1.0)

    Print sines of an array of angles given in degrees:

    >>> mg.sin(mg.tensor((0., 30., 45., 60., 90.)) * mg.pi / 180. )
    Tensor([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])
    """
    ...


@ufunc_creator(Cos)
def cos(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Trigonometric cosine, element-wise.

    This docstring was adapted from that of numpy.cos [1]_

    Parameters
    ----------
    x : ArrayLike
        Input array in radians.

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
        The corresponding cosine values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.cos.html

    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972.

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.cos([0, mg.pi/2, mg.pi])
    Tensor([  1.00000000e+00,   6.12303177e-17,  -1.00000000e+00])
    """
    ...


@ufunc_creator(Tan)
def tan(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Trigonometric tangent, element-wise.

    This docstring was adapted from that of numpy.tan [1]_

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
    y : Tensor
        The corresponding tangent values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.tan.html

    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972.

    Examples
    --------
    >>> import mygrad as mg
    >>> from math import pi
    >>> mg.tan([-pi, pi / 2, pi])
    Tensor([  1.22460635e-16,   1.63317787e+16,  -1.22460635e-16])
    """
    ...


@ufunc_creator(Arcsin)
def arcsin(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Inverse sine, element-wise.

    This docstring was adapted from that of numpy.arcsin [1]_

    Parameters
    ----------
    x : ArrayLike
        `y`-coordinate on the unit circle.

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
    angle : Tensor
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.

    See Also
    --------
    sin, cos, arccos, tan, arctan, arctan2

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arcsin` is a complex analytic function that
    has, by convention, the branch cuts [-inf, -1] and [1, inf]  and is
    continuous from above on the former and from below on the latter.

    The inverse sine is also known as `asin` or sin^{-1}.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html

    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.arcsin(1)     # pi/2
    Tensor(1.5707963267948966)
    >>> mg.arcsin(-1)    # -pi/2
    Tensor(-1.5707963267948966)
    >>> mg.arcsin(0)
    Tensor(0.0)
    """
    ...


@ufunc_creator(Arccos)
def arccos(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Inverse cosine, element-wise.

    This docstring was adapted from that of numpy.arccos [1]_

    Parameters
    ----------
    x : ArrayLike
        `x`-coordinate on the unit circle.
        For real arguments, the domain is [-1, 1].

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
    angle : Tensor
        The angle of the ray intersecting the unit circle at the given
        `x`-coordinate in radians [0, pi].

    See Also
    --------
    cos, arctan, arcsin

    Notes
    -----
    `arccos` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cos(z) = x`. The convention is to return
    the angle `z` whose real part lies in `[0, pi]`.

    For real-valued input data types, `arccos` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytic function that
    has branch cuts `[-inf, -1]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse `cos` is also known as `acos` or cos^-1.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arccos.html

    M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
    10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    We expect the arccos of 1 to be 0, and of -1 to be pi:

    >>> import mygrad as mg
    >>> mg.arccos([1, -1])
    Tensor([ 0.        ,  3.14159265])
    """
    ...


@ufunc_creator(Arctan)
def arctan(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Inverse tangent, element-wise.

    This docstring was adapted from that of numpy.arctan [1]_

    Parameters
    ----------
    x : ArrayLike

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

    See Also
    --------
    arctan2 : The "four quadrant" arctan of the angle formed by (`x`, `y`)
        and the positive `x`-axis.

    Notes
    -----
    `arctan` is a multi-valued function: for each `x` there are infinitely
    many numbers `z` such that tan(`z`) = `x`.  The convention is to return
    the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctan` is a complex analytic function that
    has [`1j, infj`] and [`-1j, -infj`] as branch cuts, and is continuous
    from the left on the former and from the right on the latter.

    The inverse tangent is also known as `atan` or tan^{-1}.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arctan.html

    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79.
    http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    We expect the arctan of 0 to be 0, and of 1 to be pi/4:

    >>> import mygrad as mg
    >>> mg.arctan([0, 1])
    Tensor([ 0.        ,  0.78539816])

    >>> mg.pi / 4
    0.78539816339744828
    """
    ...


@ufunc_creator(Arctan2)
def arctan2(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    This docstring was adapted from that of numpy.arctan [1]_

    Parameters
    ----------
    x1 : ArrayLike
        ``y``-coordinates.

    x2 : ArrayLike
        ``x``-coordinates.

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
    angle : Tensor
        Tensor of angles in radians, in the range ``[-pi, pi]``.

    See Also
    --------
    arctan, tan

    Notes
    -----
    *arctan2* is identical to the `atan2` function of the underlying
    C library.  The following special values are defined in the C
    standard: [2]_

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
     > 0   +/-inf +0 / +pi
     < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers, as are +inf
    and -inf.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.arctan.html
    .. [2] ISO/IEC standard 9899:1999, "Programming language C."

    Examples
    --------
    Consider four points in different quadrants:

    >>> import mygrad as mg
    >>> x = mg.tensor([-1.0, +1.0, +1.0, -1.0])
    >>> y = mg.tensor([-1.0, -1.0, +1.0, +1.0])
    >>> mg.arctan2(y, x) * 180 / mg.pi
    Tensor([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> mg.arctan2([1., -1.], [0., 0.])
    Tenor([ 1.57079633, -1.57079633])
    >>> mg.arctan2([0., 0., mg.inf], [+0., -0., mg.inf])
    Tenor([ 0.        ,  3.14159265,  0.78539816])"""
    ...


def sinc(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> sin(a) / a``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Sinc, a, constant=constant)


def cot(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> cot(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Cot, a, constant=constant)


def csc(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> csc(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Csc, a, constant=constant)


def sec(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> sec(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Sec, a, constant=constant)


def arccsc(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> arccsc(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccsc, a, constant=constant)


def arccot(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> arccot(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arccot, a, constant=constant)


def arcsec(a: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """``f(a) -> arcsec(a)``

    Parameters
    ----------
    a : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor"""
    return Tensor._op(Arcsec, a, constant=constant)
