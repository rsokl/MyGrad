from typing import Optional, Sequence, Union

import numpy as np

from mygrad.nnet.layers.utils import sliding_window_view
from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor

__all__ = ["Repeat"]


class Repeat(Operation):
    # Repeat can broadcast in the case:
    #    repeat(1, 2) -> [1 1]

    def __call__(
        self, a: Tensor, repeats: Union[int, Sequence[int]], axis: Optional[int] = None
    ):
        self.variables = (a,)
        self._axis = axis
        self._repeats = repeats
        return np.repeat(a.data, repeats=repeats, axis=axis)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index].data  # type: np.ndarray
        if isinstance(self._repeats, int) or len(self._repeats) == 1:
            if not isinstance(self._repeats, int):
                (self._repeats,) = self._repeats

            if not self._repeats:
                # skip accumulation if `repeats` is all zeros
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
            # We will create a grid of flat-indices commensurate with the
            # original input array. These will be used accumulate the incoming
            # gradient into the appropriate tensor elements.
            #
            # In order to deal with the flexibility of specifying multiple
            # distinct repeat values, we will perform the identical repeat
            # operation on the grid of flat-indices. Thus making them
            # commensurate with the incoming gradient. `add.at` then makes
            # short work of accumulating the incoming gradient as appropriate
            out_grad = np.zeros((a.size,), dtype=grad.dtype)
            indices = np.arange(a.size).reshape(a.shape)
            indices = np.repeat(indices, repeats=self._repeats, axis=self._axis)
            np.add.at(out_grad, indices.ravel(), grad.ravel())
            return out_grad.reshape(a.shape)
