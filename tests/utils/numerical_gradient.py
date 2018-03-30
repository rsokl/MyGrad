from decimal import Decimal

import numpy as np
from itertools import zip_longest


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


def broadcast_check(*variables):
    """ Given {a, b, ...} and the shape of op(a, b, ...), detect if any non-constant Tensor undergoes
        broadcasting via f. If so, set op.scalar_only to True, and record the broadcasted
        axes for each such tensor.

        Broadcast-incompatible shapes need not be accounted for by this function, since
        the shape of f(a, b, ...) must already be known.

        Parameters
        ----------
        variables : Sequence[ArrayLike]

        Returns
        -------
        Tuple[Dict[str, Tuple[int, ...]]
            ({'new_axes': (...), 'keepdim_axes: (...)}, ...)
            For each variable, indicates which, if any, axes need be summed over
            to reduce the broadcasted gradient for back-prop through that variable."""
    variables = variables
    out_shape = np.broadcast(*variables).shape
    new_axes = [[] for i in range(len(variables))]
    keepdims = [[] for i in range(len(variables))]

    # no broadcasting occurs for non-constants
    if all(var.shape == out_shape for var in variables):
        return tuple(dict(new_axes=tuple(new), keepdim_axes=tuple(keep))
                     for new, keep in zip(new_axes, keepdims))

    # check size of aligned dimensions
    for n, dims in enumerate(zip_longest(*(var.shape[::-1] for var in variables))):
        axis = len(out_shape) - 1 - n
        if len(set(i for i in dims if (i is not None))) <= 1:
            continue

        for var_index, i in enumerate(dims):
            # broadcasting occurs over existing dim: e.g. (2,1,5) w/ (2,3,5) -> (2,3,5)
            if i == 1:
                keepdims[var_index].append(axis)

    for var_index, var in enumerate(variables):
        keepdims[var_index] = tuple(keepdims[var_index])

        # a new axis is created to allow broadcasting: e.g. (3,) w/ (2,3) -> (2,3)
        if var.ndim < len(out_shape):
            new_axes[var_index] = tuple(range(len(out_shape) - var.ndim))
    return tuple(dict(new_axes=tuple(new), keepdim_axes=tuple(keep))
                 for new, keep in zip(new_axes, keepdims))


def broadcast_back(grad, new_axes, keepdim_axes):
    """ Sum-reduce df/dx, where f was produced by broadcasting x along
        the broadcasting axes. This assumes that that the gradient of a scalar
        is ultimately being computed. """
    if keepdim_axes:
        grad = grad.sum(axis=keepdim_axes, keepdims=True)

    if new_axes:
        grad = grad.sum(axis=new_axes)

    return grad

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


def numerical_gradient(f, *args, back_grad, vary_ind=None, h=1e-8):
    """ Computes numerical partial derivatives of f(x0, x1, ...) in each
        of its variables, using the central difference method.

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
    args = tuple(to_decimal_array(arr) for arr in args)

    # get axis & keepdims args for collapsing a broadcasted gradient
    all_broad_args = broadcast_check(*args)

    grads = [None]*len(args)

    def gen_fwd_diff(i):
        # x1, ..., x_i + h, ..., xn
        return ((var if j != i else var + h) for j, var in enumerate(args))

    def gen_bkwd_diff(i):
        # x1, ..., x_i - h, ..., xn
        return ((var if j != i else var - h) for j, var in enumerate(args))

    for n, broad_args in enumerate(all_broad_args):
        if vary_ind is not None and n not in vary_ind:
            continue
        # central difference in variable n
        dvar = (f(*gen_fwd_diff(n)) - f(*gen_bkwd_diff(n))) / (Decimal(2) * h)
        grads[n] = broadcast_back(back_grad * dvar.astype(float), **broad_args)

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


def numerical_gradient_full(f, *, x, back_grad,  h=1e-8):
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
        f_fwd = to_decimal_array(f(x_fwd.astype(float)))

        x_bkwd = x_fwd
        x_bkwd[ind] -= Decimal(2) * h
        f_bkwd = to_decimal_array(f(x_bkwd.astype(float)))

        dxi = to_decimal_array((f_fwd - f_bkwd) / (Decimal(2) * h))
        grad[ind] = (dxi.astype('float') * back_grad).sum()
    return grad.astype(float)
