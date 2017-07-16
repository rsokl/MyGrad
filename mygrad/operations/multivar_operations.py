from itertools import zip_longest

__all__ = ["MultiVarOperation",
           "MultiVarBroadcastableOp"]


class MultiVarOperation:
    """ Experimental! Permits arbitrary number of tensor operands."""
    scalar_only = False

    def __call__(self, *input_vars):
        self.variables = input_vars
        return NotImplementedError

    def backward_var(self, grad, index):
        raise NotImplementedError

    def backward(self, grad):
        """ Back-propagates the gradient through all of the operation's inputs. This needs to be updated
            by an operation if that operation takes more than 2 Tensor arguments."""
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for attr in self.__dict__:
            var = getattr(self, attr)
            if hasattr(var, 'null_gradients'):
                var.null_gradients()


class MultiVarBroadcastableOp(MultiVarOperation):
    """ Experimental! Permits arbitrary number of tensor operands.
        A subclass of Operation that allows for back-propagation through broadcasted operations.

        If broadcasting occurs with a non-constant tensor, then MyGrad's back-propagation system
        requires that the computational graph's terminal node, which triggers the back-propagation,
        is a scalar.

        Broadcastable operations must run `broadcast_check` during __call__.
        (see `Add` for an example)"""

    def broadcast_check(self, *variables, out_shape):
        """ Given {a, b, ...} and the shape of op(a, b, ...), detect if any non-constant Tensor undergoes
            broadcasting via f. If so, set op.scalar_only to True, and record the broadcasted
            axes for each such tensor.

            Broadcast-incompatible shapes need not be accounted for by this function, since
            the shape of f(a, b, ...) must already be known.

            Parameters
            ----------
            variables : Sequence[mygrad.Tensor
            out_shape : Sequence[int]
                The shape of f(a, b)."""
        self.variables = variables
        self.new_axes = [[] for i in range(len(variables))]
        self.keepdims = [[] for i in range(len(variables))]

        # no broadcasting occurs for non-constants
        if all((var.constant or var.shape == out_shape) for var in variables):
            return None

        # check size of aligned dimensions
        for n, dims in enumerate(zip_longest(*(var.shape[::-1] for var in self.variables))):
            axis = len(out_shape) - 1 - n
            if len(set(i for i in dims if (i is not None))) <= 1:
                continue

            for var_index, i in enumerate(dims):
                # broadcasting occurs over existing dim: e.g. (2,1,5) w/ (2,3,5) -> (2,3,5)
                if i == 1:
                    self.keepdims[var_index].append(axis)

        for var_index, var in enumerate(self.variables):
            self.keepdims[var_index] = tuple(self.keepdims[var_index])

            if not var.constant:
                # a new axis is created to allow broadcasting: e.g. (2,) w/ (2,3) -> (2,3)
                if var.ndim < len(out_shape):
                    self.new_axes[var_index] = tuple(range(len(out_shape) - var.ndim))

                if self.new_axes[var_index] or self.new_axes[var_index]:
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

    def backward_var(self, grad, index):
        return self.broadcast_back(grad, self.new_axes[index], self.keepdims[index])
