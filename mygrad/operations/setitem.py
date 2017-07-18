from .operation_base import BroadcastableOp
import numpy as np


class SetItem(BroadcastableOp):
    @staticmethod
    def broadcast_back(grad, new_axes, keepdim_axes):
        """ Sum-reduce df/dx, where f was produced by broadcasting x along
            the broadcasting axes. This assumes that that the gradient of a scalar
            is ultimately being computed. """
        if keepdim_axes:
            grad = grad.sum(axis=keepdim_axes, keepdims=True)

        if new_axes:
            grad = grad.sum(axis=new_axes)

        return grad

    def broadcast_single(self, a, b_shape, out_shape):
        """ Given a, in_shape, and the shape of op(a, in_shape), detect if any non-constant Tensor undergoes
            broadcasting via f. If so, set op.scalar_only to True, and record the broadcasted
            axes for each such tensor.

            Broadcast-incompatible shapes need not be accounted for by this function, since
            the shape of f(a, in_shape) must already be known.

            Parameters
            ----------
            a : pygrad.Tensor
            b_shape : Sequence[int]
            out_shape : Sequence[int]
                The shape of f(a, in_shape)."""
        new_axes = []
        keepdims = []

        # no broadcasting occurs for non-constants
        if a.constant or (a.shape == out_shape):
            return new_axes, keepdims

        # check size of aligned dimensions
        for n, (i, j) in enumerate(zip(a.shape[::-1], b_shape[::-1])):
            axis = len(out_shape) - 1 - n
            if i != j and i == 1:
                # broadcast over existing dim
                keepdims.append(axis)

        # broadcasting into new dims
        if a.ndim < len(out_shape):
            new_axes = tuple(range(len(out_shape) - a.ndim))

        if new_axes or keepdims:
            self.scalar_only = True
        return new_axes, tuple(keepdims)

    def __call__(self, a, b, index):

        self.a = a
        self.b = b

        self.index = index
        a.data[index] = b.data

        if not self.b.constant:
            op_shape = np.asarray(a.data[index]).shape
            self.new_axes_b, self.keepdims_b = self.broadcast_single(self.b, op_shape, op_shape)
        return a.data

    def backward_a(self, grad):
        grad = np.copy(grad)
        grad[self.index] = 0
        self.a.backward(grad)

    def backward_b(self, grad):
        grad = super(SetItem, self).backward_b(grad[self.index])

        # handle the edge case of "projecting down" on setitem. E.g:
        # x = Tensor([0, 1, 2])
        # y = Tensor([3])
        # x[0] = y  # this is legal since x[0] and y have the same size
        if grad.ndim < self.b.ndim:
            grad = grad.reshape(self.b.shape)
        self.b.backward(grad)
