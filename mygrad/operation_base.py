__all__ = ["Operation",
           "BroadcastableOp"]


class Operation:
    """ Base class for all tensor operations that support back-propagation
        of gradients."""
    scalar_only = False

    def __call__(self, *input_vars):
        """ Performs a forward pass, f, of this Operation:
                  f(x1, ...., xn) -> out

            Parameters
            ----------
            *input_vars : mygrad.Tensor
                The input-arguments of f. The tuple (x1, ...., xn)
                should be bound to the instance-attribute `self.variables`

            Returns
            -------
            numpy.ndarray
                The output of the forward pass function."""

        self.variables = input_vars
        raise NotImplementedError

    def backward_var(self, grad, index, **kwargs):
        """ Given ``grad = d(out)/d(f)``, computes ``d(out)/d(var)``, and passes this result
            to ``var.backward()``, where var is the tensor-argument at position ``index``.

            Parameters
            ----------
            grad : numpy.ndarray
                The back-propagated total derivative with respect to the present
                operation (`f`): d(out)/df

            index : int
                The index-location of ``var`` in ``self.variables``

            Raises
            ------
            NotImplemented Error"""
        raise NotImplementedError

    def backward(self, grad, **kwargs):
        """ Back-propagates the gradient through all of the operation's inputs.
            Constant tensors do not propagate a gradient."""
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index, **kwargs)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input tensors,
            and to all preceding tensors in the computational graph."""
        for var in self.variables:
            var.null_gradients()


class BroadcastableOp(Operation):
    """ Signals that an Operation's forward pass can broadcast its tensor
        arguments."""
    pass
