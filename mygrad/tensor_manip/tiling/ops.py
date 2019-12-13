from typing import Optional, Sequence, Union

import numpy as np

from mygrad.nnet.layers.utils import sliding_window_view
from mygrad.operation_base import BroadcastableOp
from mygrad.tensor_base import Tensor

__all__ = ["Repeat"]


class Repeat(BroadcastableOp):
    # Repeat can broadcast in the case:
    # repeat(1, 2) -> [1 1]
    def __call__(
        self, a: Tensor, repeats: Union[int, Sequence[int]], axis: Optional[int] = None
    ):
        self.variables = (a,)
        self._axis = axis
        self._repeats = repeats
        return np.repeat(a.data, repeats=repeats, axis=axis)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index].data  # type: np.ndarray
        if isinstance(self._repeats, int):
            if not self._repeats:
                return np.zeros(a.shape, dtype=grad.dtype)

            if self._axis is None:
                # input array was treated as if it was flattened
                grad = grad.ravel()
                window_shape = (self._repeats,)
            else:
                # if `a` is a scalar, we will treat it like a
                # 1D array, since the `repeat` op did broadcasting
                window_shape = [1] * max(1, a.ndim)
                window_shape[self._axis] = self._repeats
                window_shape = tuple(window_shape)

            # Create windowed view of gradient, where each window
            # extends/strides along the repeated axis, and with a
            # window size given by `repeats`. Thus summing over the
            # trailing window dimensions accumulates the gradient
            # to the appropriate "source" entries of the input tensor
            grad = sliding_window_view(
                grad, window_shape=window_shape, step=window_shape
            )
            grad = grad.sum(axis=tuple(range(-len(window_shape), 0)))

            if self._axis is None:
                grad.shape = a.shape
            return grad
        else:
            raise NotImplementedError()
