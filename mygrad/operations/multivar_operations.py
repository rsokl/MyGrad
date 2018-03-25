__all__ = ["MultiVarOperation",
           "MultiVarBroadcastableOp"]


class MultiVarOperation:
    """ Experimental! Permits arbitrary number of tensor operands."""
    scalar_only = False

    def __call__(self, *input_vars):
        self.variables = input_vars
        return NotImplementedError

    def backward_var(self, grad, index, **kwargs):
        raise NotImplementedError

    def backward(self, grad, **kwargs):
        """ Back-propagates the gradient through all of the operation's inputs. This needs to be updated
            by an operation if that operation takes more than 2 Tensor arguments."""
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index, **kwargs)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for var in self.variables:
            var.null_gradients()


class MultiVarBroadcastableOp(MultiVarOperation):
    """ Experimental! Permits arbitrary number of tensor operands.
        A subclass of Operation that allows for back-propagation through broadcasted operations.
        If broadcasting occurs with a non-constant tensor, then MyGrad's back-propagation system
        requires that the computational graph's terminal node, which triggers the back-propagation,
        is a scalar.
        Broadcastable operations must run `broadcast_check` during __call__.
        (see `Add` for an example)"""
    pass
