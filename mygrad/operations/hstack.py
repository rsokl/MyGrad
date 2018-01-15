from .multivar_operations import MultiVarOperation
import numpy as np
from itertools import accumulate, zip_longest


class Hstack(MultiVarOperation):
    def __call__(self, *input_vars):
        for i, dim in enumerate(list(zip_longest(*[var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if (i == 0 and input_vars[0].data.ndim == 1) or (i == 1):
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.axis = 0 if input_vars[0].data.ndim is 1 else 1
        self.indices = list(accumulate([var.data.shape[0] if self.axis is 0 else var.data.shape[1] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.hstack([var.data for var in input_vars])
        return out

    def backward(self, grad):
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def backward_var(self, grad, index):
        var = self.variables[index]
        if not self.axis:
            var.backward(grad[self.indices[index]:self.indices[index+1]])
        else:
            var.backward(grad[:,self.indices[index]:self.indices[index+1]])
