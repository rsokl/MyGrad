from mygrad.operation_base import Operation, BroadcastableOp
import numpy as np

__all__ = ["MatMul"]


class MatMul(BroadcastableOp):
    scalar_only = True
    def __call__(self, a, b):
        """ f(a) -> matmul(a, b)

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor

            Returns
            -------
            numpy.ndarray"""
        self.variables = (a, b)
        return np.matmul(a.data, b.data)

    def backward_var(self, grad, index, **kwargs):
        a, b = (i.data for i in self.variables)

        # handle 1D w/ 1D (dot product of vectors)
        if a.ndim == 1 and b.ndim == 1:
            if index == 0:
                dfdx = grad * b
            elif index == 1:
                dfdx = grad * a
            
            self.variables[index].backward(dfdx, **kwargs)
            return
        
        if index == 0:  # compute grad through a
            if b.ndim > 1:  # ([...], j) w/ ([...], j, k)
                if a.ndim == 1:
                    grad = np.expand_dims(grad, -2)
                dfdx = np.matmul(grad, b.swapaxes(-1, -2))
            else:           # ([...], i, j) w/ (j,)
                dfdx = np.expand_dims(grad, -1) * b
        
        if index == 1:  # compute grad through b
            if a.ndim > 1:  # ([...], i, j) w/ ([...], j, [k])
                if b.ndim == 1:
                    grad = np.expand_dims(grad, -1)
                dfdx = np.matmul(a.swapaxes(-1, -2), grad)
                if b.ndim == 1:
                    dfdx = dfdx.squeeze(-1)
            else:           # (j,) w/ ([...], j, k)
                dfdx = a[:, np.newaxis] * np.expand_dims(grad, -2)
            
        self.variables[index].backward(dfdx, **kwargs)
