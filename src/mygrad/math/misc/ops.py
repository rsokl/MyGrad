from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from mygrad.operation_base import BinaryUfunc, UnaryUfunc

__all__ = ["Abs", "Sqrt", "Cbrt", "Maximum", "Minimum"]


class Abs(UnaryUfunc):
    numpy_ufunc = np.absolute

    def __call__(self, *args, nan_to_num: bool = True, **kwargs):
        self._nan_to_num = nan_to_num
        return super().__call__(*args, **kwargs)

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables

        return grad * np.piecewise(
            a.data,
            [a.data < 0, a.data == 0, a.data > 0],
            [-1, (0 if self._nan_to_num else np.nan), 1],
        )


class Sqrt(UnaryUfunc):
    numpy_ufunc = np.sqrt

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (2 * np.sqrt(a.data))


class Cbrt(UnaryUfunc):
    numpy_ufunc = np.cbrt

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (3 * np.cbrt(a.data ** 2))


class _MaxMin(BinaryUfunc, ABC):
    @staticmethod
    @abstractmethod
    def _comparison(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Returns `True` where elements of the output were derived from `x1`.

        I.e. max => x1 > x2  ;  min => x1 < x2"""
        raise NotImplementedError()

    def __init__(self):
        super().__init__()
        self._where_tensor_a_was_selected: Optional[np.ndarray] = None

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables

        if self._where_tensor_a_was_selected is None:
            self._where_tensor_a_was_selected = self._comparison(a.data, b.data)

        if index == 0:
            mask = self._where_tensor_a_was_selected
        elif index == 1:
            equal_mask = a.data == b.data
            mask = np.logical_not(self._where_tensor_a_was_selected)

            if mask.ndim:
                np.logical_not(mask, out=mask, where=equal_mask)
            elif equal_mask:
                mask = np.logical_not(mask)
        else:  # pragma: no cover
            raise IndexError(f"Back-propagation through tensor-{index}")

        return mask * grad


class Maximum(_MaxMin):
    numpy_ufunc = np.maximum
    _comparison = staticmethod(np.greater)


class Minimum(_MaxMin):
    numpy_ufunc = np.minimum
    _comparison = staticmethod(np.less)


class MatMul(BinaryUfunc):
    numpy_ufunc = np.matmul
    _supports_where = False

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        a = a.data
        b = b.data

        # handle 1D w/ 1D (dot product of vectors)
        if a.ndim == 1 and b.ndim == 1:
            if index == 0:
                return grad * b
            elif index == 1:
                return grad * a

        if index == 0:  # compute grad through a
            if b.ndim > 1:  # ([...], j) w/ ([...], j, k)
                if a.ndim == 1:
                    grad = np.expand_dims(grad, -2)
                dfdx = np.matmul(grad, b.swapaxes(-1, -2))
            else:  # ([...], i, j) w/ (j,)
                dfdx = np.expand_dims(grad, -1) * b
            return dfdx

        if index == 1:  # compute grad through b
            if a.ndim > 1:  # ([...], i, j) w/ ([...], j, [k])
                if b.ndim == 1:
                    grad = np.expand_dims(grad, -1)
                dfdx = np.matmul(a.swapaxes(-1, -2), grad)
                if b.ndim == 1:
                    dfdx = dfdx.squeeze(-1)
            else:  # (j,) w/ ([...], j, k)
                dfdx = a[:, np.newaxis] * np.expand_dims(grad, -2)
            return dfdx
        else:  # pragma: no cover
            raise ValueError()
