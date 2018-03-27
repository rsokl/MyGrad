# from mygrad.operations.multivar_operations import MultiVarOperation
# from .operation_base import Operation
# import numpy as np
#
#
# class GetItem(Operation):
#     def __call__(self, a, index):
#         self.a = a
#         self.index = index
#         out = self.a.data[index]
#         self.shape = out.shape if isinstance(out, np.ndarray) else None
#         return out
#
#     def backward_a(self, grad):
#         out = np.zeros_like(self.a.data)
#         out[self.index] = grad
#         self.a.backward(out)


from mygrad.operations.multivar_operations import Operation
import numpy as np


class GetItem(Operation):
    def __call__(self, a, index):
        self.variables = (a,)
        self.index = index
        out = a.data[index]
        self.shape = out.shape if isinstance(out, np.ndarray) else None
        return out

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        out = np.zeros_like(a.data)
        out[self.index] = grad
        a.backward(out, **kwargs)
