__all__ = ["MultiVarOperation",
           "MultiVarBroadcastableOp"]


class MultiVarOperation:
    """ Base class for all tensor operations that support backprop.

        Functions accept `Tensor` objects and return Python numeric types """
    scalar_only = False

    def __call__(self, *input_vars):
        """ An operation instance, `f`, performs a forward pass using this function. Typically,
            it is called in this form:
                f(a, b) -> out

            Where `a` and `b` are Tensor-instances, and `out` is a Numpy-array.

            It must also bind `a` and `b` to the operation instance:
                self.a = a
                self.b = b"""

        self.variables = input_vars
        return NotImplementedError

    def backward_var(self, grad, index, **kwargs):
        """ Given grad = d(L)/d(f), computes d(L)/d(var), and passes this result to var.backward(),
            where var is the tensor-argument at position `index`

            Parameters
            ----------
            grad : numpy.ndarray
                The back-propagated total derivative with respect to the present operation (`f`): dL/df

            Raises
            ------
            NotImplemented Error"""
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
