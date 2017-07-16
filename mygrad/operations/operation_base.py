import numpy as np

__all__ = ["Operation",
           "BroadcastableOp"]


class Operation:
    """ Base class for all tensor operations that support backprop.

        Functions accept `Tensor` objects and return Python numeric types """

    scalar_only = False

    def __call__(self, *args, **kwargs):
        """ An operation instance, `f`, performs a forward pass using this function. Typically,
            it is called in this form:
                f(a, b) -> out

            Where `a` and `b` are Tensor-instances, and `out` is a Numpy-array.

            It must also bind `a` and `b` to the operation instance:
                self.a = a
                self.b = b"""
        raise NotImplementedError

    def backward_a(self, grad):
        """ Given grad = d(L)/d(f), computes d(L)/d(a), and passes this result to a.backward():
                a._backward( dL/da )

            Parameters
            ----------
            grad : numpy.ndarray
                The back-propagated total derivative with respect to the present operation (`f`): dL/df

            Raises
            ------
            NotImplemented Error"""
        raise NotImplementedError

    def backward_b(self, grad):
        """ Given grad = d(L)/d(f), computes d(L)/d(b), and passes this result to b.backward():
                b._backward( dL/da )

            Parameters
            ----------
            grad : numpy.ndarray
                The back-propagated total derivative with respect to the present operation (`f`): dL/df

            Raises
            ------
            NotImplemented Error"""
        raise NotImplementedError

    def backward(self, grad):
        """ Back-propagates the gradient coming from the operation's output, back to its inputs."""
        if not self.a.constant:
            self.backward_a(grad)
        if hasattr(self, 'b') and not self.b.constant:
            self.backward_b(grad)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for attr in self.__dict__:
            var = getattr(self, attr)
            if hasattr(var, 'null_gradients'):
                var.null_gradients()


class BroadcastableOp(Operation):
    def broadcast_check(self, a, b, out_shape):
        """ Given a, b, and the shape of op(a, b), detect if any non-constant Tensor undergoes
            broadcasting via f. If so, set op.scalar_only to True, and record the broadcasted
            axes for each such tensor.

            Broadcast-incompatible shapes need not be accounted for by this function, since
            the shape of f(a, b) must already be known.

            Parameters
            ----------
            a : pygrad.Tensor
            b : pygrad.Tensor
            out_shape : Sequence[int]
                The shape of f(a, b)."""
        self.a = a
        self.b = b

        self.new_axes_a = []
        self.new_axes_b = []

        self.keepdims_a = []
        self.keepdims_b = []

        # no broadcasting occurs for non-constants
        if (a.constant or (a.shape == out_shape)) and (b.constant or (b.shape == out_shape)):
            return None

        # check size of aligned dimensions
        for n, (i, j) in enumerate(zip(self.a.shape[::-1], self.b.shape[::-1])):
            axis = len(out_shape) - 1 - n
            if i != j:
                if i == 1:
                    # broadcasting occurs over existing dim: e.g. (2,1) w/ (2,3) -> (2,3)
                    self.keepdims_a.append(axis)
                else:
                    self.keepdims_b.append(axis)

        self.keepdims_a = tuple(self.keepdims_a)
        self.keepdims_b = tuple(self.keepdims_b)

        if not self.a.constant:
            # a new axis is created to allow broadcasting: e.g. (2,) w/ (2,3) -> (2,3)
            if self.a.ndim < len(out_shape):
                self.new_axes_a = tuple(range(len(out_shape) - self.a.ndim))

            if self.new_axes_a or self.keepdims_a:
                self.scalar_only = True

        if not self.b.constant:
            if self.b.ndim < len(out_shape):
                self.new_axes_b = tuple(range(len(out_shape) - self.b.ndim))

            if self.new_axes_b or self.keepdims_b:
                self.scalar_only = True

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

    def backward_a(self, grad):
        return self.broadcast_back(grad, self.new_axes_a, self.keepdims_a)

    def backward_b(self, grad):
        return self.broadcast_back(grad, self.new_axes_b, self.keepdims_b)