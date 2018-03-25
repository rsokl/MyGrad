from mygrad.operations.multivar_operations import MultiVarOperation


class Reshape(MultiVarOperation):
    def __call__(self, a, shape):
        self.variables = (a,)
        return a.data.reshape(shape)

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        a.backward(grad.reshape(*a.shape), **kwargs)
