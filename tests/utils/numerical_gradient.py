from decimal import Decimal

import numpy as np
from itertools import zip_longest

from mygrad._utils import reduce_broadcast


def to_decimal_array(arr):
    """ Convert numpy ND-array to Decimal-type object array of the same shape.
        Used for facilitating high-precision arithmetic.

        Parameters
        ----------
        arr : Union[float, numpy.ndarray]

        Returns
        -------
        numpy.ndarray
        Decimal-type object array"""
    arr = np.asarray(arr)
    return np.array(tuple(Decimal(float(i)) for i in arr.flat), dtype=Decimal).reshape(arr.shape)


def numerical_derivative(f, x, h=1e-8):
    """ Computes the numerical derivate of f(x) at `x`::

                  dfdx = (f(x + h) - f(x - h)) / (2h)

        Makes use of `decimal.Decimal` for high-precision arithmetic.

        Parameters
        ----------
        f : Callable[[Real], Real]
            A unary function: f(x)

        x : Decimal
            The value at at which the derivative is computed

        h : Real, optional (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        Decimal
            df/dx @ `x` """

    h = Decimal(h)
    dx = (Decimal(f(x + h)) - Decimal(f(x - h))) / (Decimal(2) * h)
    return dx


def numerical_gradient(f, *args, back_grad, vary_ind=None, h=1e-8, as_decimal=True, kwargs={}):
    """ Computes numerical partial derivatives of f(x0, x1, ...) in each
        of its variables, using the central difference method.

        This is a "fast" method - it varies entire arrays at once. Thus
        this is only appropriate for trivial vectorized functions that
        map accross entries of arrays (like add or multiply). E.g.
        matrix multiplication is *not* suited for this style of gradient.

        Parameters
        ----------
        f : Callable[[numpy.ndarray, ...], numpy.ndarray]
            f(x, ...) -> numpy.ndarray

        *args : Tuple[numpy.ndarray, ...]
            The input arguments to be fed to f.

        back_grad : numpy.ndarray
            The gradient being back-propagated to x and y, via f

        vary_ind : Optional[Tuple[int, ...]]
            If `None`, the partials of f with respect to all the inputs are.
            computed. Otherwise you can specify a sequence of the indices
            of the variables whose partials are to be computed
               0 -> w.r.t x only, 1 -> w.r.t y only, etc.

        h : float, optional, (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        Tuple[Union[NoneType, numpy.ndarray], ...]
            df/dx0, df/dx1, ... - evaluated at (`x0`, `x1`, ... ).
        """

    if not args:
        raise ValueError("At least one value must be passed to `args`")

    h = Decimal(h)
    args = tuple(to_decimal_array(i) if as_decimal else i for i in args)

    grads = [None]*len(args)

    def gen_fwd_diff(i):
        # x1, ..., x_i + h, ..., xn
        return ((var if j != i else var + h) for j, var in enumerate(args))

    def gen_bkwd_diff(i):
        # x1, ..., x_i - h, ..., xn
        return ((var if j != i else var - h) for j, var in enumerate(args))

    for n in range(len(args)):
        if vary_ind is not None and n not in vary_ind:
            continue
        # central difference in variable n
        dvar = (f(*gen_fwd_diff(n), **kwargs) - f(*gen_bkwd_diff(n), **kwargs)) / (Decimal(2) * h)
        grads[n] = reduce_broadcast(back_grad * dvar.astype(float), args[n].shape)

    return grads


def numerical_gradient_sequence(f, *, x, back_grad,  axis=None, keepdims=False, h=1e-8,
                                no_axis=False, no_keepdims=False):
    """ Computes numerical partial derivatives of f({x}), where f is a numpy-style
        sequential function (e.g. numpy.sum, numpy.mean, ...). The partial derivative
        is computed for each member of {x}

        Parameters
        ----------
        f : Callable[[numpy.ndarray], numpy.ndarray]
            f({x}) -> numpy.ndarray

        x : numpy.ndarray
            An array storing the sequence(s) of values in the array. More than once
            sequence may be designated, according to the `axis` argument of `f`.

        axis : Optional[None, int, Tuple[int, ...]]
            The value of the `axis` argument, to be fed to `f`.

        keepdims : bool, optional (default=False)
            The value of the `keepdims` argument, to be fed to `f`.

        back_grad : numpy.ndarray
            The gradient being back-propagated to {x}, via f

        h : float, optional, (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        numpy.ndarray
            df/d{x}
        """

    grad = np.empty_like(x)
    x = to_decimal_array(x)
    h = Decimal(h)

    kwargs = dict()
    if not no_axis:
        kwargs["axis"] = axis
    if not no_keepdims:
        kwargs["keepdims"] = keepdims

    for ind, val in np.ndenumerate(x):
        x_fwd = np.copy(x)
        x_fwd[ind] += h
        f_fwd = f(x_fwd, **kwargs)

        x_bkwd = x_fwd
        x_bkwd[ind] -= Decimal(2) * h
        f_bkwd = f(x_bkwd, **kwargs)

        dxi = to_decimal_array((f_fwd - f_bkwd) / (Decimal(2) * h))
        grad[ind] = (dxi.astype('float') * back_grad).sum()
    return grad.astype(float)


def numerical_gradient_full(f, *args, back_grad, as_decimal=True, kwargs={}, vary_ind=None):
    """ Computes numerical partial derivatives of f(x, y, ..., **kwargs), by
        varying each entry of x, y, ... independently producing a gradient
        in each variable.

        Parameters
        ----------
        f : Callable[[numpy.ndarray], numpy.ndarray]
            f(x) -> numpy.ndarray

        x : numpy.ndarray
            An array storing the sequence(s) of values in the array. More than once
            sequence may be designated, according to the `axis` argument of `f`.

        back_grad : numpy.ndarray
            The gradient being back-propagated to {x}, via f

        h : float, optional, (default=1e-8)
            Approximating infinitesimal.

        as_decimal : bool, optional (default=False)
            If False, x is passed to f as a Decimal-type array. This
            improves numerical precision, but is not permitted by some functions.

        kwargs : Dict[str, Any], optional (default=dict())
            The keyword arguments to be passed to f.

        vary_ind : Optional[Tuple[int, ...]]
            If `None`, the partials of f with respect to all the inputs are.
            computed. Otherwise you can specify a sequence of the indices
            of the variables whose partials are to be computed
               0 -> w.r.t x only, 1 -> w.r.t y only, etc.

        Returns
        -------
        Tuple[numpy.ndarray, ...]
            df/dx, df/dy, ...
            df/dvar will be None if var was excluded via `vary_ind`
        """

    args = tuple(to_decimal_array(i) if as_decimal else i for i in args)
    grads = [None] * len(args)
    if isinstance(vary_ind, int):
        vary_ind = [vary_ind]

    for n in range(len(args)):
        if vary_ind is not None and n not in vary_ind:
            continue
        tmp_f = lambda var: f(*args[:n], var, *args[n+1:], **kwargs)
        grads[n] = _numerical_gradient_full(tmp_f,
                                            x=args[n],
                                            back_grad=back_grad,
                                            as_decimal=as_decimal)
    return tuple(grads)


def _numerical_gradient_full(f, *, x, back_grad, h=1e-8, as_decimal=False):
    """ Computes numerical partial derivatives of f(x), by
        varying each entry of `x` independently.

        Parameters
        ----------
        f : Callable[[numpy.ndarray], numpy.ndarray]
            f(x) -> numpy.ndarray

        x : numpy.ndarray
            An array storing the sequence(s) of values in the array. More than once
            sequence may be designated, according to the `axis` argument of `f`.

        back_grad : numpy.ndarray
            The gradient being back-propagated to {x}, via f

        h : float, optional, (default=1e-8)
            Approximating infinitesimal.

        as_decimal : bool, optional (default=False)
            If False, x is passed to f as a Decimal-type array. This
            improves numerical precision, but is not permitted by some functions.

        Returns
        -------
        numpy.ndarray
            df/dx
        """

    grad = np.empty_like(x)
    x = to_decimal_array(x)
    h = Decimal(h)

    for ind, val in np.ndenumerate(x):
        x_fwd = np.copy(x)
        x_fwd[ind] += h
        f_fwd = to_decimal_array(f(x_fwd) if as_decimal else f(x_fwd.astype(float)))

        x_bkwd = x_fwd
        x_bkwd[ind] -= Decimal(2) * h
        f_bkwd = to_decimal_array(f(x_bkwd) if as_decimal else f(x_bkwd.astype(float)))

        df_dxi = to_decimal_array((f_fwd - f_bkwd) / (Decimal(2) * h))
        grad[ind] = (df_dxi.astype('float') * back_grad).sum()
    return grad.astype(float)
