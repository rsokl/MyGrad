from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import numpy as np

from mygrad._utils.graph_tracking import TRACK_GRAPH
from mygrad.operation_base import Operation

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor

__all__ = ["Concatenate"]


class Concatenate(Operation):
    def __call__(
        self,
        *input_vars: "Tensor",
        axis: Optional[int] = 0,
        out: Optional[np.ndarray] = None,
        dtype=None,
    ) -> np.ndarray:
        kwargs = {"out": out, "axis": axis}
        if dtype is not None:
            # compatibility shim: min numpy 1.20
            kwargs["dtype"] = dtype

        out = np.concatenate(tuple(var.data for var in input_vars), **kwargs)

        self.variables = tuple(input_vars)
        if TRACK_GRAPH:
            self.axis = axis
            if axis is not None:
                # need to make sure axis is non-negative so that
                # axis checking during backprop is simplified
                self.axis = axis % out.ndim
                self.indices = list(
                    accumulate(var.data.shape[axis] for var in input_vars)
                )
                self.indices.insert(0, 0)
            else:
                # inputs were flattened
                self.indices = np.cumsum((0,) + tuple(var.size for var in input_vars))
        return out

    def backward_var(self, grad, index, **kwargs) -> np.ndarray:
        var = self.variables[index]
        if self.axis is None:
            return grad[self.indices[index] : self.indices[index + 1]].reshape(
                var.shape
            )

        return grad[
            tuple(
                slice(None, None, None)
                if dim != self.axis
                else slice(self.indices[index], self.indices[index + 1])
                for dim in range(var.data.ndim)
            )
        ]


class Stack(Operation):
    def __call__(
        self,
        *input_vars: "Tensor",
        axis: int = 0,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = np.stack(tuple(var.data for var in input_vars), axis=axis, out=out)

        self.variables = tuple(input_vars)

        if TRACK_GRAPH:
            # need to make sure axis is non-negative so that
            # axis checking during backprop is simplified
            self.axis = axis % out.ndim

        return out

    def backward_var(self, grad, index, **kwargs) -> np.ndarray:
        assert 0 <= index
        var = self.variables[index]

        if var.data.ndim == 0:
            return grad[index]

        return grad[
            tuple(
                slice(None, None, None) if dim != self.axis else index
                for dim in range(var.data.ndim + 1)
            )
        ]
