from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from typing import Optional, Tuple, Union

import numpy as np

from mygrad.operation_base import Sequential

__all__ = [
    "CumProd",
    "CumSum",
    "Max",
    "Mean",
    "Min",
    "Prod",
    "StdDev",
    "Sum",
    "Variance",
]


class MaxMin(Sequential, ABC):
    @staticmethod
    @abstractmethod
    def _arg_finder(
        a: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> np.ndarray:
        """`numpy.argmax` or `numpy.argmin` - in correspondence with the
        implementation of `max` or `min`."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def numpy_func(
        self,
        a: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError()

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        axis = self.axis

        if a.ndim == 0:
            return grad

        if hasattr(axis, "__iter__"):
            axis = tuple(ax % a.ndim for ax in axis)
            axis = None if len(axis) == a.ndim else tuple(sorted(axis))
        elif axis is not None:
            axis = (axis % a.ndim,)

        # normalize shape of grad to be same as when keepdims=False
        if self.keepdims:
            if axis is not None:
                reduce_ = [slice(None)] * a.ndim
                for i in axis:
                    reduce_[i] = 0
                reduce_ = tuple(reduce_)
            else:
                reduce_ = (0,) * a.ndim
            grad = grad[reduce_]

        # max(a) -> use argmax
        if axis is None:
            indices = np.unravel_index(self._arg_finder(a.data), a.shape)
            out = np.zeros_like(a.data, dtype=grad.dtype)
            out[indices] = grad
            return out
        # max(x, axis=i) -> use argmax with specified axis
        elif len(axis) == 1:
            op_index = self._arg_finder(a.data, axis=axis[0])
            indices = list(np.indices(op_index.shape))
            indices.insert(axis[0], op_index)
            indices = tuple(indices)

            out = np.zeros_like(a.data, dtype=grad.dtype)
            out[indices] = grad
            return out

        # max(x, axis=(i,j,...) ) -> Reshape data to use argmax along trailing axis
        static_ax = tuple(
            sorted(set(range(a.ndim)) - set(axis))
        )  # non-reduced axes (m, n, ..)
        to_trans = static_ax + axis  # (m, n, ..., i, j, ...)
        from_trans = tuple(np.argsort(to_trans))
        outshape = tuple(a.shape[i] for i in static_ax)

        z = a.data.transpose(*to_trans).reshape(*outshape, -1)  # (m, n, ..., i*j*[...])

        k = self._arg_finder(z, axis=-1)
        indices = tuple(i for i in np.indices(k.shape))
        indices += (k,)
        tmp_grad_shape = z.shape

        out = np.zeros(tmp_grad_shape, dtype=grad.dtype)
        out[indices] = grad
        shape = tuple(a.shape[i] for i in to_trans)
        return out.reshape(shape).transpose(*from_trans)


class Max(MaxMin):
    numpy_func = staticmethod(np.max)
    _arg_finder = staticmethod(np.argmax)


class Min(MaxMin):
    numpy_func = staticmethod(np.min)
    _arg_finder = staticmethod(np.argmin)


class Sum(Sequential):
    numpy_func = staticmethod(np.sum)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        if self.axis is None:
            return np.full(a.shape, grad, dtype=a.dtype)

        if not self.keepdims:
            index = [slice(None) for i in range(a.ndim)]
            for i in self.axis:
                index[i] = np.newaxis
            grad = grad[tuple(index)]
        return np.broadcast_to(grad, a.shape).astype(a.dtype, copy=False)


class Mean(Sum):
    numpy_func = staticmethod(np.mean)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        n = (
            a.data.size
            if self.axis is None
            else np.prod([a.shape[i] for i in self.axis])
        )
        return super().backward_var(grad / n, index, **kwargs)


class Prod(Sequential):
    numpy_func = staticmethod(np.prod)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        x = a.data

        if a.ndim == 0:
            return grad

        axes = (
            set(range(a.ndim))
            if self.axis is None
            else {i if i >= 0 else a.ndim + i for i in self.axis}
        )

        # make grad broadcast-compatible against x
        grad = grad.reshape(*(1 if n in axes else i for n, i in enumerate(a.shape)))

        # This is a valid method for taking the derivative
        # of prod(x) only if there are no zeros in `x`.
        # If there are zeros we need to patch some of the nans
        # that we just created, with the correct derivative.
        with np.errstate(divide="ignore", invalid="ignore"):
            dldx = np.prod(x, axis=self.axis, keepdims=True) / x

        # if two or more zeros occur within a given sequence, then
        # the nans in the sequences can simply be set to 0
        if np.any(np.isnan(dldx)):
            x = x.copy()

            # computes the number of 0s to occur within each sequence
            has_zero = np.broadcast_to(
                np.sum(x == 0, axis=self.axis, keepdims=True), x.shape
            )
            dldx[has_zero > 1] = np.nan_to_num(dldx[has_zero > 1])

            # if only a single 0 occurs within a given sequence, the
            # derivative needs to be recomputed at that location by
            # setting that element 0 -> 1
            if np.any(np.isnan(dldx)):
                is_zero = x == 0
                x[is_zero] = 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    loc = np.logical_and(is_zero, has_zero == 1)
                    dldx[loc] = (np.prod(x, axis=self.axis, keepdims=True) / x)[loc]

        return grad * dldx


def _reverse_cumsum(
    x: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:  # pragma: no cover
    """ (x0, x1, x2) -> (x0, x0 + x1, x0 + x1 + x2)"""
    if axis is None:
        axis = 0
    return np.flip(np.cumsum(np.flip(x, axis=axis), axis=axis), axis=axis)


def _find_first_zeros_along_axis(x, axis):  # pragma: no cover
    """Return the indices at which 0 first occurs in `x` as viewed
    along the specified axis

    Parameters
    ----------
    x : numpy.ndarray
    axis : Union[None, int]
        The axis along which zeros are looked for. If
        `None`, then `x` must be a flat-array

    Returns
    -------
    Tuple[Tuple[int, ...], ...]
        x.ndim tuple-entries, specifying the corresponding
        positions where the first 0 is encountered along the
        given axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 1, 0],
                      [1, 1, 0],
                      [1, 1, 0]])

    All of the zeros fall along a single column, thus
    only one is "encountered" when looking across rows
    within each column for a single zero. It is located
    at: row-0, col-2

    >>> _find_first_zeros_along_axis(x, axis=0)
    ((0,), (2,))

    Looking along the columns within each row,
    each of the three zeros are "found", they are
    located at: row-0, col=2
                row-1, col=2
                row-2, col=2

    >>> _find_first_zeros_along_axis(x, axis=1)
    ((0, 1, 2), (2, 2, 2))"""

    def add_to_seen(seen, inds):
        if inds[1:] not in (i[1:] for i in seen):
            seen.append(inds)
        return seen

    def move(seq, origin, dest):
        if origin == dest:
            return seq
        o = seq.pop(origin)
        seq.insert(dest, o)
        return seq

    wer = np.where((np.moveaxis(x, axis, 0) if axis is not None else x) == 0)

    if axis is None:
        axis = 0

    gen_inds = (
        move(list(seq), origin=0, dest=axis)
        for seq in reduce(add_to_seen, zip(*wer), [])
    )
    return tuple(zip(*gen_inds))


class CumProd(Sequential):
    numpy_func = staticmethod(np.cumprod)
    _integer_axis_only = True

    def backward_var(self, grad, index, **kwargs):
        x = self.variables[index].data
        axis = self.axis
        g = grad

        if axis is None:
            orig_shape = x.shape
            x = x.flat
            g = g.flat
        else:
            orig_shape = None
            if axis < 0:
                axis += x.ndim

        g_cumprod = g * np.cumprod(x, axis=axis)

        # This is a valid method for taking the derivative
        # of cumprod(x) only if there are no zeros in `x`.
        # If there are zeros we need to patch some of the nans
        # that we just created, with the correct derivative.
        with np.errstate(divide="ignore", invalid="ignore"):
            # assuming x0, ..., xn are all non-zero
            #
            # dldx = [g0 + g1*x1 + g2*x1*x2 + ...,
            #              g1*x0 + g2*x0*x2 + ...,
            #                      g2*x0*x1 + ...,
            #                               + ...]
            dldx = _reverse_cumsum(g_cumprod, axis=axis) / x

        # Only the first occurrences of 0 along the specified
        # axis in x need have its correct derivative computed
        # instead of being nan. All other nans in `dldx` can be
        # safely set to zero since they fall "downstream" from
        # a 0 in the cumulative-product and the derivatives of
        # all such elements are 0.
        # See `_find_first_zeros_along_axis` for more details
        if np.any(np.isnan(dldx)):
            x = x.copy()
            locs = _find_first_zeros_along_axis(x, axis=axis)
            x[locs] = 1

            g_cumprod = g * np.cumprod(x, axis=axis)
            with np.errstate(divide="ignore", invalid="ignore"):
                dldx[locs] = (_reverse_cumsum(g_cumprod, axis=axis) / x)[locs]
            np.nan_to_num(dldx, copy=False)

        if axis is None:
            dldx.shape = orig_shape
        return dldx


class CumSum(Sequential):
    _integer_axis_only = True
    numpy_func = staticmethod(np.cumsum)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        g = _reverse_cumsum(grad, self.axis)
        if self.axis is None:
            g.shape = a.shape
        return g


class Variance(Sequential):
    numpy_func = staticmethod(np.var)

    def _grad_preprocess(self, grad: np.ndarray) -> np.ndarray:
        """Helper method provided so that `Variance` and `StdDev` can
        share the same implementation for `backward_var`."""
        return grad

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        axis: Optional[Tuple[int, ...]] = self.axis
        if isinstance(axis, Sequence) and len(axis) == 0:
            return np.zeros(a.shape, dtype=float)

        N = a.size if axis is None else np.prod([a.shape[i] for i in axis])
        N -= self.ddof

        grad = self._grad_preprocess(grad)
        if grad.ndim == 0:
            # keepdims=False and axis=None (or all axes)
            grad = np.full(a.shape, grad, dtype=float)
        else:
            axis: Tuple[int, ...]
            if not self.keepdims:
                index = [slice(None)] * a.ndim
                for i in axis:
                    index[i] = np.newaxis
                grad = grad[tuple(index)]
        back = (2.0 / N) * (a.data - a.data.mean(axis=axis, keepdims=True))
        return back * grad


class StdDev(Variance):
    numpy_func = staticmethod(np.std)

    def _grad_preprocess(self, grad: np.ndarray) -> np.ndarray:
        """Helper method provided so that `Variance` and `StdDev` can
        share the same implementation for `backward_var`.

        Includes backpropagation through the sqrt after the variance."""
        (a,) = self.variables
        return grad / (
            2
            * np.sqrt(
                a.data.var(axis=self.axis, ddof=self.ddof, keepdims=self.keepdims)
            )
        )
