from collections.abc import Sequence
from functools import reduce
from typing import Any

import numpy as np

from mygrad.operation_base import Operation

__all__ = ["MaxMin", "Sum", "Mean", "Prod", "CumProd", "CumSum", "Variance", "StdDev"]


class MaxMin(Operation):
    def __call__(self, a, axis=None, keepdims=False, maxmin=None):
        """ Return the maximum (minimum) of a tensor, or along its axes.

            Parameters
            ----------
            a : pygrad.Tensor
                Input data.

            axis : Optional[int, Tuple[int, ...]]
                Axis or axes along which to operate. By default, flattened input is used.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `arr`.

            maxmin : str
                'max' or 'min'. Selects the operation that is performed

            Returns
            -------
            amax : ndarray
                Maximum (minimum) of `a`. If `axis` is None, the result is a 0-D array."""
        assert maxmin in ("max", "min"), "Invalid keyword argument"
        op = np.argmax if maxmin == "max" else np.argmin

        # let numpy handle error checking
        np.amax(np.empty([1] * a.ndim), axis=axis, keepdims=keepdims)

        self.variables = (a,)

        if a.ndim == 0:
            return a.data

        if hasattr(axis, "__iter__"):
            assert isinstance(axis, tuple)
            axis = tuple(ax % a.ndim for ax in axis)
            axis = None if len(axis) == a.ndim else tuple(sorted(axis))
        elif axis is not None:
            axis = (axis % a.ndim,)

        self.axis = axis
        self.keepdims = keepdims

        # max(a) -> use argmax
        if self.axis is None:
            self.indices = np.unravel_index(op(a.data), a.shape)
            dat = a.data[self.indices]

        # max(x, axis=i) -> use argmax with specified axis
        elif len(self.axis) == 1:  #
            op_index = op(a.data, axis=self.axis[0])
            self.indices = list(np.indices(op_index.shape))
            self.indices.insert(self.axis[0], op_index)
            self.indices = tuple(self.indices)
            dat = a.data[self.indices]

        # max(x, axis=(i,j,...) ) -> Reshape data to use argmax along trailing axis
        else:
            self.static_ax = tuple(
                sorted(set(range(a.ndim)) - set(self.axis))
            )  # non-reduced axes (m, n, ..)
            self.to_trans = self.static_ax + self.axis  # (m, n, ..., i, j, ...)
            self.from_trans = tuple(np.argsort(self.to_trans))
            outshape = tuple(a.shape[i] for i in self.static_ax)

            z = a.data.transpose(*self.to_trans).reshape(
                *outshape, -1
            )  # (m, n, ..., i*j*[...])

            k = op(z, axis=-1)
            self.indices = tuple(i for i in np.indices(k.shape))
            self.indices += (k,)
            self.tmp_grad_shape = z.shape
            z = z[self.indices]

            dat = z.reshape(outshape)  # (m, n, ...)

        if not self.keepdims:
            return dat

        elif self.axis is None:
            keep_index = (np.newaxis,) * a.ndim
        else:
            keep_index = [slice(None)] * a.ndim
            for i in self.axis:
                keep_index[i] = np.newaxis
            keep_index = tuple(keep_index)

        return np.asarray(dat)[keep_index]

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        if a.ndim == 0:
            return grad

        # normalize shape of grad to be same as when keepdims=False
        if self.keepdims:
            if self.axis is not None:
                reduce = [slice(None)] * a.ndim
                for i in self.axis:
                    reduce[i] = 0
                reduce = tuple(reduce)
            else:
                reduce = (0,) * a.ndim
            grad = grad[reduce]

        # use argmax indices to broadcast grad to correct elements
        if self.axis is None or len(self.axis) == 1:
            out = np.zeros_like(a.data, dtype=float)
            out[self.indices] = grad
            return out
        else:
            out = np.zeros(self.tmp_grad_shape, dtype=float)
            out[self.indices] = grad
            shape = tuple(a.shape[i] for i in self.to_trans)
            return out.reshape(shape).transpose(*self.from_trans)


class Sum(Operation):
    def __call__(self, a, axis=None, keepdims=False):
        """ Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        self.axis = axis

        self.keepdims = keepdims
        out = a.data.sum(axis=axis, keepdims=keepdims)
        self.outshape = out.shape if isinstance(out, np.ndarray) else None
        return out

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        if self.outshape is None:
            return np.full(a.shape, grad, dtype=float)

        if not self.keepdims:
            index = [slice(None) for i in range(a.ndim)]
            for i in self.axis:
                index[i] = np.newaxis
            grad = grad[tuple(index)]
        return np.broadcast_to(grad, a.data.shape).astype(float)


class Mean(Sum):
    def __call__(self, a, axis=None, keepdims=False):
        out = super().__call__(a, axis, keepdims)
        self.n = (
            a.data.size
            if self.axis is None
            else np.prod([a.shape[i] for i in self.axis])
        )
        return out / self.n

    def backward_var(self, grad, index, **kwargs):
        return super().backward_var(grad / self.n, index, **kwargs)


class Prod(Operation):
    def __call__(self, a, axis=None, keepdims=False):
        """ Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)
        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        self.axis = axis
        self.keepdims = keepdims
        return a.data.prod(axis=axis, keepdims=keepdims)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        x = a.data
        grad = np.asarray(grad)

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
            print("HERE")

            # computes the number of 0s to occur within each sequence
            has_zero = np.broadcast_to(
                np.sum(x == 0, axis=self.axis, keepdims=True), x.shape
            )
            dldx[has_zero > 1] = np.nan_to_num(dldx[has_zero > 1])

            # if only a single 0 occurs within a given sequence, the
            # derivative needs to be recomputed at that location by
            # setting that element 0 -> 1
            if np.any(np.isnan(dldx)):
                print("HERE2")
                is_zero = x == 0
                x[is_zero] = 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    loc = np.logical_and(is_zero, has_zero == 1)
                    dldx[loc] = (np.prod(x, axis=self.axis, keepdims=True) / x)[loc]

        return grad * dldx


def _reverse_cumsum(x, axis=None):  # pragma: no cover
    """ (x0, x1, x2) -> (x0, x0 + x1, x0 + x1 + x2)"""
    if axis is None:
        axis = 0
    return np.flip(np.cumsum(np.flip(x, axis=axis), axis=axis), axis=axis)


def _find_first_zeros_along_axis(x, axis):  # pragma: no cover
    """ Return the indices at which 0 first occurs in `x` as viewed
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


class CumProd(Operation):
    def __call__(self, a, axis=None):
        self.variables = (a,)
        self.axis = axis
        return np.cumprod(a.data, axis)

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


class CumSum(Operation):
    def __call__(self, a, axis=None):
        self.variables = (a,)
        self.axis = axis
        return np.cumsum(a.data, axis)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        g = _reverse_cumsum(grad, self.axis)
        if self.axis is None:
            g.shape = a.shape
        return g


class Variance(Operation):
    _method_name = "var"

    def _grad_preprocess(self, grad: Any) -> np.ndarray:
        """Helper method provided so that `Variance` and `StdDev` can
        share the same implementation for `backward_var`."""
        return np.asarray(grad)

    def __call__(self, a, axis=None, keepdims=False, ddof=0):
        self.variables = (a,)

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)

        self.kwargs = dict(axis=axis, keepdims=keepdims, ddof=ddof)
        return getattr(a.data, self._method_name)(**self.kwargs)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        if isinstance(self.kwargs["axis"], Sequence) and len(self.kwargs["axis"]) == 0:
            return np.zeros(a.shape, dtype=float)

        N = (
            a.size
            if self.kwargs["axis"] is None
            else np.prod([a.shape[i] for i in self.kwargs["axis"]])
        )
        N -= self.kwargs["ddof"]

        grad = self._grad_preprocess(grad)
        if grad.ndim == 0:
            grad = np.full(a.shape, grad, dtype=float)
        else:
            if not self.kwargs["keepdims"]:
                index = [slice(None)] * a.ndim
                for i in self.kwargs["axis"]:
                    index[i] = np.newaxis
                grad = grad[tuple(index)]
        back = (2.0 / N) * (
            a.data - a.data.mean(axis=self.kwargs["axis"], keepdims=True)
        )
        return back * grad


class StdDev(Variance):
    _method_name = "std"

    def _grad_preprocess(self, grad: Any) -> np.ndarray:
        """Helper method provided so that `Variance` and `StdDev` can
        share the same implementation for `backward_var`.

        Includes backpropagation through the sqrt after the variance."""
        a = self.variables[0]
        return np.asarray(grad) / (2 * np.sqrt(a.data.var(**self.kwargs)))
