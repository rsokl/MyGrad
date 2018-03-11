from decimal import Decimal
import numpy as np
from itertools import zip_longest


def to_decimal_array(arr):
   return np.array(tuple(Decimal(float(i)) for i in arr.flat), dtype=Decimal).reshape(arr.shape)


def broadcast_check(*variables, out_shape):
    """ Given {a, b, ...} and the shape of op(a, b, ...), detect if any non-constant Tensor undergoes
        broadcasting via f. If so, set op.scalar_only to True, and record the broadcasted
        axes for each such tensor.
        Broadcast-incompatible shapes need not be accounted for by this function, since
        the shape of f(a, b, ...) must already be known.

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor]
        out_shape : Sequence[int]
            The shape of f(a, b)."""
    variables = variables
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


def numerical_gradient(f, *, x, y, back_grad, vary_ind=None, h=1e-8):
    """ Computes numerical partial derivatives of f(x, y)

        Parameters
        ----------
        f : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            f(x, y) -> numpy.ndarray

        x : numpy.ndarray

        y : numpy.ndarray

        back_grad : numpy.ndarray
            The gradient being back-propagated to x and y, via f

        vary_ind : Optional[0 or 1]
            If `None`, partials of f with respect to x and y, respectively, are
            both computes. Otherwise 0 -> w.r.t x only, 1 -> w.r.t y only.

        h : float, optional, (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        Tuple[Union[NoneType, numpy.ndarray], Union[NoneType, numpy.ndarray]]
            dfdx, dfdy - both evaluated at `x` and `y`.
        """
    if vary_ind not in {None, 0, 1}:
        raise ValueError("`vary_ind` must be `None`, `0`, or `1`. Passed {}".format(vary_ind))

    h = Decimal(h)
    x = to_decimal_array(x)
    y = to_decimal_array(y)

    outshape = np.broadcast(x, y).shape
    x_args, y_args = broadcast_check(x, y, out_shape=outshape)
    dx, dy, = None, None
    if vary_ind is None or vary_ind == 0:
        dx = (f(x + h, y) - f(x - h, y)) / (Decimal(2) * h)
        dx = broadcast_back(back_grad * dx.astype(float), **x_args)
    if vary_ind is None or vary_ind == 1:
        dy = (f(x, y + h) - f(x, y - h)) / (Decimal(2) * h)
        dy = broadcast_back(back_grad * dy.astype(float), **y_args)

    return dx, dy


