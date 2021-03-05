import numpy as np

from mygrad.operation_base import Operation


class Where(Operation):
    def __call__(self, a, b, *, condition):
        self.variables = (a, b)
        self.condition = np.asarray(condition, dtype=bool)
        return np.where(condition, a.data, b.data)

    def backward_var(self, grad, index, **kwargs):
        condition = self.condition if index == 0 else ~self.condition
        return np.where(condition, grad, 0)
