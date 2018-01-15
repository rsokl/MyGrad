from .multivar_operations import MultiVarOperation
import numpy as np
from itertools import accumulate, zip_longest


class Dstack(MultiVarOperation):
    def __call__(self, *input_vars):
        for i, dim in enumerate(list(zip_longest(*[var.data[np.newaxis,:,np.newaxis].shape if len(var.data.shape) is 1 else var.data[...,np.newaxis].shape if len(var.data.shape) is 2 else var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if i == 2:
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.indices = list(accumulate([1 if len(var.data.shape) is 1 else 1 if len(var.data.shape) is 2 else var.data.shape[2] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.dstack([var.data for var in input_vars])

        return out

    def backward(self, grad):
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def backward_var(self, grad, index):
        var = self.variables[index]
        if len(var.data.shape) == 1:
            var.backward(np.squeeze(grad[:,:,self.indices[index]:self.indices[index+1]]))
        elif len(var.data.shape) == 2:
            var.backward(np.squeeze(grad[:,:,self.indices[index]:self.indices[index+1]], axis=2))
        else:
            var.backward(grad[:,:,self.indices[index]:self.indices[index+1]])
