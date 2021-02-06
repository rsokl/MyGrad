import numpy as np

from mygrad.operation_base import BinaryArith, UnaryArith

__all__ = ["Abs", "Sqrt", "Cbrt", "Maximum", "Minimum"]


class Abs(UnaryArith):
    numpy_ufunc = np.absolute

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad * np.piecewise(
            a.data, [a.data < 0, a.data == 0, a.data > 0], [-1, np.nan, 1]
        )


class Sqrt(UnaryArith):
    numpy_ufunc = np.sqrt

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (2 * np.sqrt(a.data))


class Cbrt(UnaryArith):
    numpy_ufunc = np.cbrt

    def backward_var(self, grad, index, **kwargs):
        (a,) = self.variables
        return grad / (3 * np.cbrt(a.data ** 2))


class Maximum(BinaryArith):
    numpy_ufunc = np.maximum

    def __init__(self):
        super().__init__()
        self.greater_than_mask = None

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables

        if self.greater_than_mask is None:
            self.greater_than_mask = a.data > b.data

        if index == 0:
            mask = self.greater_than_mask
        elif index == 1:
            equal_mask = a.data == b.data
            mask = np.logical_not(self.greater_than_mask)

            if mask.ndim:
                np.logical_not(mask, out=mask, where=equal_mask)
            elif equal_mask:
                mask = np.logical_not(mask)
        else:  # pragma: no cover
            raise IndexError(f"Back-propagation through tensor-{index}")

        return mask * grad


class Minimum(BinaryArith):
    numpy_ufunc = np.minimum

    def __init__(self):
        super().__init__()
        self.less_than_mask = None
        self.equal_mask = None

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables

        if self.less_than_mask is None:
            self.less_than_mask = a.data < b.data

        if index == 0:
            mask = self.less_than_mask
        elif index == 1:
            mask = np.logical_not(self.less_than_mask)
            equal_mask = a.data == b.data

            if mask.ndim:
                np.logical_not(mask, out=mask, where=equal_mask)
            elif equal_mask:
                mask = np.logical_not(mask)
        else:  # pragma: no cover
            raise IndexError(f"Back-propagation through tensor-{index}")

        return mask * grad
