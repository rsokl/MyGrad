from .multivar_operations import MultiVarOperation
import numpy as np
from itertools import accumulate, zip_longest


class Stack(MultiVarOperation):
    def __call__(self, *input_vars, axis=0):
        assert len({var.data.shape for var in input_vars}) == 1, "all input Tensors must have the same shape"

        self.variables = input_vars
        self.axis = axis
        newaxis = [slice(None, None, None) if dim is not axis else None for dim in range(len(input_vars[0].data.shape)+1)]
        self.indices = list(accumulate([var.data[newaxis].shape[axis] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.stack([var.data for var in input_vars], axis=axis)

        return out

    def backward(self, grad):
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def backward_var(self, grad, index):
        var = self.variables[index]
        grad_slice = [slice(None, None, None) if dim is not self.axis else slice(self.indices[index], self.indices[index+1]) for dim in range(len(var.data.shape)+1)]
        var.backward(np.squeeze(grad[grad_slice], axis=self.axis))
