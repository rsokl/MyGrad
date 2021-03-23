from collections import Counter
from copy import copy
from functools import reduce
from itertools import chain
from numbers import Real
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided

from mygrad._utils import SkipGradient, reduce_broadcast
from mygrad._utils.graph_tracking import TRACK_GRAPH
from mygrad.operation_base import Operation

__all__ = ["EinSum", "Norm"]


# EinSum #


def _unique_from_end(in_str):
    """Return a string with all redundant characters removed,
    removing left-most redundant entries

    i.e. "ijikik" -> "jik"

    Parameters
    ----------
    in_str: str

    Returns
    -------
    str

    Examples
    --------
    >>> _unique_from_end("ijikik")
    "jik"
    """

    return reduce(lambda acc, x: acc + x if x not in acc else acc, in_str[::-1], "")[
        ::-1
    ]


def _merge_max_mappings(*mappings):
    """Merge dictionaries based on largest values in key->value.

    Parameters
    ----------
    *mappings : Dict[Any, Any]

    Returns
    -------
    Dict[Any, Any]

    Examples
    --------
    >>> _merge_max_mappings({"a":1, "b":4}, {"a":2})
    {"a":2, "b":4}
    """

    def _merge_max(d1, d2):
        d1.update((k, v) for k, v in d2.items() if d1.get(k, 0) < v)
        return d1

    return reduce(_merge_max, mappings, {})


def _get_indices(item, seq):
    """Return the indices where `item` occurs in `seq`

    Returns
    -------
    Generator[int]"""
    return (n for n, x in enumerate(seq) if x == item)


class EinSum(Operation):
    can_return_view = True

    def __call__(self, *variables, in_lbls, out_lbls, out=None, optimize=False):
        """
        einsum('{in_lbls}->{out_lbls}', *variables, optimize=optimize)

        Parameters
        ----------
        variables : mygrad.Tensor
        in_lbls : str
        out_lbls : str
        optimize : bool

        Returns
        -------
        numpy.ndarray
        """
        self.in_lbls = in_lbls.split(",")
        self.out_lbls = out_lbls
        self.variables = variables
        self.optimize = optimize

        # cache counts the number of redundant tensor-label pairs
        # fed to einsum. Only one gradient will be computed for a
        # unique tensor-label pair
        self._cache = None

        # einsum doesn't handle out=None properly in numpy 1.17
        kwargs = {} if out is None else {"out": out}
        return np.einsum(
            "->".join((in_lbls, out_lbls)),
            *(var.data for var in self.variables),
            optimize=optimize,
            **kwargs,
        )

    @property
    def cache(self) -> Counter:
        if self._cache is None:
            # This is hacky, but because this caching mechanism depends on the tensor-ids,
            # we have to build the cache here - in case einsum is used to produce a view
            # involved in an inplace operations, and placeholder tensors need be replaced.
            #
            # Creating the cache in __call__ could create a nasty inconsistency between
            # tensor ids
            self._cache = Counter(zip((id(v) for v in self.variables), self.in_lbls))
        return self._cache

    def backward_var(self, grad, index, **kwargs):
        """
        example
        -------
        fwd:          "ijk, k -> ji", x, y
        bkwd (var: 0): "ji, k -> ijk", grad, y
        bkwd (var: 1): "ji, ijk -> k", grad, x
        """

        # ijk, k
        in_lbls = copy(self.in_lbls)
        original_var_lbl = in_lbls.pop(index)
        var = self.variables[index]

        factor = self.cache[(id(var), original_var_lbl)]
        if factor == 0:
            # the gradient for the current tensor-label pair
            # has already been computed, scaled, and back-propped,
            # skip gradient calculation.
            raise SkipGradient()

        numpy_arrays = tuple(i.data for i in self.variables)
        self.cache[(id(var), original_var_lbl)] = 0

        var_lbl = _unique_from_end(original_var_lbl)
        repeat_lbls = len(var_lbl) != len(original_var_lbl)

        if repeat_lbls:
            # example fwd-prop: einsum("iji -> ij", x)
            # "iji" becomes "ji", later we will write along
            # the diagonal of an array to reinstate this axis that
            # we just removed
            mapping_gen = (
                {k: v for k, v in zip(lbl, arr.shape)}
                for lbl, arr in zip(self.in_lbls, numpy_arrays)
            )
            lbl_to_size = _merge_max_mappings(*mapping_gen)
            var_shape = tuple(lbl_to_size[lbl] for lbl in var_lbl)
        else:
            var_shape = self.variables[index].shape

        # ji
        grad_lbl = self.out_lbls

        # Catch indices over which un-contracted sum was performed
        # for the given variable: e.g for var-0 in "ijk, jk -> k"
        # i is summed over without contraction with another tensor
        #
        # Backpropping through this is illegal, as it requires the creation
        # of an axis; e.g. k, jk -> ijk
        # Broadcast the gradient along all such dimensions; e.g. k -> ik
        # then proceed as usual; e.g. ik, jk -> ijk
        unique_in_lbls = set(chain.from_iterable(in_lbls)) | set(grad_lbl)
        if len(set(var_lbl) - unique_in_lbls) > 0:
            exp_dims = [slice(None) for i in range(grad.ndim)]
            grad_shape = list(grad.shape)
            for n, lbl in enumerate(var_lbl):
                if lbl not in unique_in_lbls:
                    grad_lbl = grad_lbl[:n] + lbl + grad_lbl[n:]
                    exp_dims.insert(n, np.newaxis)
                    grad_shape.insert(n, var_shape[n])

            grad = np.broadcast_to(
                grad if not grad.ndim else grad[tuple(exp_dims)], grad_shape
            )

        # "ji, k -> ijk"
        back_prop_lbls = ",".join([grad_lbl] + in_lbls) + "->" + var_lbl

        # (grad, y)
        operands = (grad,) + numpy_arrays[:index] + numpy_arrays[index + 1 :]

        if not repeat_lbls:
            # dfdx: einsum("ji, k -> ijk", grad, y)
            outshape = self.variables[index].shape
            dfdx = reduce_broadcast(
                np.einsum(back_prop_lbls, *operands, optimize=self.optimize), outshape
            )
            if var_shape != dfdx.shape:
                # if y was broadcast over x, the gradient needs to
                # be broadcast to x's shape: dfdx-shape (i,j,1) -> (i,j,k)
                dfdx = np.broadcast_to(dfdx, var_shape)
            if factor > 1:
                # This tensor-label pair appears several times as
                # input to einsum. Scale the gradient accordingly
                # such that the full contribution of the tensor-label
                # pair is accounted for.
                dfdx *= factor
            return dfdx

        # Accommodate trace by writing to strided view on array of zeros
        # For example:
        #
        # fwd:  einsum('ijkji, k -> jk', x, y)
        # dfdx: einsum('jk, k -> kji', grad, y, out=view_of_x)
        #
        # writing to `view_of_x`, which is a view along the appropriate
        # diagonals of x, is equivalent to:
        #
        # dfdx: einsum('jk, k -> ijkji', grad, y)
        #
        # which is formally correct but not supported by einsum.
        dfdx = np.zeros(tuple(lbl_to_size[i] for i in original_var_lbl))
        out_view_shape = tuple(lbl_to_size[i] for i in var_lbl)

        # compute strides required to traverse the appropriate diagonals of
        # the output tensor.
        strides = tuple(
            sum(dfdx.strides[ind] for ind in _get_indices(lbl, original_var_lbl))
            for lbl in var_lbl
        )
        out_view = as_strided(dfdx, shape=out_view_shape, strides=strides)
        np.einsum(back_prop_lbls, *operands, out=out_view, optimize=self.optimize)
        if factor > 1:
            # This tensor-label pair appears several times as
            # input to einsum. Scale the gradient accordingly
            # such that the full contribution of the tensor-label
            # pair is accounted for.
            dfdx *= factor
        return dfdx


class Norm(Operation):
    def __call__(self, tensor, ord=None, axis=None, keepdims=False):
        self.variables = (tensor,)
        out = np.linalg.norm(tensor.data, ord=ord, axis=axis, keepdims=keepdims)

        if isinstance(ord, Real) and np.isinf(ord):  # pragma: no cover
            raise NotImplementedError(
                "inf norms should be handled by mygrad.max(abs(x))"
            )

        if (tensor.ndim == 2 and ord is not None and axis is None) or (
            hasattr(axis, "__len__") and len(axis) > 1
        ):
            raise NotImplementedError(
                "mygrad.linalg.norm does not support matrix norms"
            )

        if ord == 0:
            raise NotImplementedError(
                "mygrad.linalg.norm(..., ord=0) is not a differentiable operation"
            )

        if TRACK_GRAPH:
            if hasattr(axis, "__len__"):
                (axis,) = axis

            self.axis: Optional[int] = axis
            self.keepdims = keepdims
            self.ord = ord if ord is not None else 2

            # self._norm is broadcast-compatible with `tensor`
            if self.keepdims is False:
                _axis = (
                    self.axis if self.axis is not None else tuple(range(tensor.ndim))
                )

                self._norm: np.ndarray = np.expand_dims(out, axis=_axis)
            else:
                self._norm: np.ndarray = out

        return out

    def backward_var(self, grad: np.ndarray, index: int, **kwargs) -> np.ndarray:
        (tensor,) = self.variables
        x = tensor.data

        if self.keepdims is False:
            # is broadcast-compatible with `tensor`
            _axis = self.axis if self.axis is not None else tuple(range(tensor.ndim))
            grad = np.expand_dims(grad, axis=_axis)

        invalid_derivative = np.where(x == 0)

        if self.ord == 1:
            out = np.sign(x)
            out *= grad

        elif self.ord == 2:
            out = x / self._norm
            out *= grad

        else:
            out = np.fabs(x)
            if out.size:
                # if out.size is 0, then this produces div-by-zero
                _norm = self._norm / np.sum(
                    out ** self.ord, axis=self.axis, keepdims=True
                )
            else:
                _norm = +self._norm
            out **= self.ord - 1
            out *= np.sign(x)
            out *= _norm
            out *= grad
        out[invalid_derivative] = np.nan
        return out
