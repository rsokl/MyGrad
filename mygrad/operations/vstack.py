from .multivar_operations import MultiVarOperation
import numpy as np
from itertools import accumulate, zip_longest


class Vstack(MultiVarOperation):
    def __call__(self, *input_vars):
        for i, dim in enumerate(list(zip_longest(*[var.data[np.newaxis,np.newaxis].shape if not var.data.ndim else var.data[np.newaxis].shape if var.data.ndim is 1 else var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if i == 0:
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.indices = list(accumulate([1 if (0,1).__contains__(var.data.ndim)  else var.data.shape[0] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.vstack([var.data for var in input_vars])

        return out

    def backward(self, grad):
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def backward_var(self, grad, index):
        var = self.variables[index]
        if not var.data.ndim:
            var.backward(np.asscalar(grad[self.indices[index]:self.indices[index+1]]))
        elif var.data.ndim == 1:
            var.backward(np.squeeze(grad[self.indices[index]:self.indices[index+1]], axis=0))
        else:
            var.backward(grad[self.indices[index]:self.indices[index+1]])
