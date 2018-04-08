from mygrad.operation_base import Operation, BroadcastableOp
import numpy as np


__all__ = ["GetItem",
           "SetItem"]


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
        np.add.at(out, self.index, grad)
        a.backward(out, **kwargs)


def arr(*shape): return np.arange(np.prod(shape)).reshape(shape)


class SetItem(BroadcastableOp):

    def __call__(self, a, b, index):
        """ a[index] = b

            Parameters
            ----------
            a : mygrad.Tensor
            b : mygrad.Tensor
            index : valid-array-index"""
        self.variables = (a, b)

        self.index = index if isinstance(index, tuple) else (index,)
        a.data[index] = b.data

        return a.data

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            grad = np.copy(grad)
            grad[self.index] = 0
            kwargs["_broadcastable"] = False
            a.backward(grad, **kwargs)
        elif index == 1:
            grad_sel = np.asarray(grad[self.index])

            if not np.shares_memory(grad_sel, grad) and grad_sel.size > 0 and grad_sel.ndim > 0:
                if len(self.index) > 1 or not np.issubdtype(np.asarray(self.index[0]).dtype, np.bool_):
                    unique = arr(*grad.shape)
                    sub_sel = unique[self.index].flat
                    elements, first_inds, = np.unique(np.flip(sub_sel, axis=0), return_index=True)
                    if len(first_inds) < len(sub_sel):
                        first_inds = (len(sub_sel) - 1) - first_inds
                        mask = np.zeros_like(sub_sel)
                        mask[first_inds] = 1
                        mask = mask.reshape(grad_sel.shape)
                        grad_sel *= mask

            # handle the edge case of "projecting down" on setitem. E.g:
            # x = Tensor([0, 1, 2])
            # y = Tensor([3])
            # x[0] = y  # this is legal since x[0] and y have the same size
            if grad_sel.ndim < b.ndim:
                grad_sel = grad_sel.reshape(b.shape)
            b.backward(grad_sel, **kwargs)
        else:
            raise IndexError

