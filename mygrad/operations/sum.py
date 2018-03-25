from mygrad.operations.multivar_operations import MultiVarOperation
import numpy as np


__all__ = ["Sum", "Mean"]


class Sum(MultiVarOperation):

    def __call__(self, a, axis=None, keepdims=False):
        """ Parameters
            ----------
            a : mygrad.Tensor"""
        self.variables = (a,)

        if axis is not None and not hasattr(axis, "__iter__"):
            axis = (axis,)
        self.axis = axis

        self.keepdims = keepdims
        out = a.data.sum(axis=axis, keepdims=keepdims)
        self.outshape = out.shape if isinstance(out, np.ndarray) else None
        return out
    
    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        if self.outshape is None:
            a.backward(np.full(a.shape, grad, dtype=float))
            return None

        if not self.keepdims:
            index = [slice(None) for i in range(a.ndim)]
            for i in self.axis:
                index[i] = np.newaxis
            grad = grad[index]

        a.backward(np.broadcast_to(grad, a.data.shape).astype(float), **kwargs)


class Mean(Sum):
    def __call__(self, a, axis=None, keepdims=False):
        out = super(Mean, self).__call__(a, axis, keepdims)
        self.n = a.data.size if not self.axis else np.prod([a.shape[i] for i in self.axis])
        return out / self.n

    def backward_var(self, grad, index, **kwargs):
        super(Mean, self).backward_var(grad / self.n, index, **kwargs)
