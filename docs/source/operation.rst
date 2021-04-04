Writing Your Own Operations
***************************

Let's write our own "multiply" operation.

.. code:: python

   import numpy as np

   import mygrad as mg
   from mygrad import prepare_op
   from mygrad.operation_base import Operation
   from mygrad.typing import ArrayLike

   # All operations should inherit from Operation, or one of its subclasses
   class CustomMultiply(Operation):
       """ Performs f(x, y) = x * y """

       def __call__(self, x: mg.Tensor, y: mg.Tensor) -> np.ndarray:
           # This method defines the "forward pass" of the operation.
           # It must bind the variable tensors to the op and compute
           # the output of the operation as a numpy array

           # All tensors must be bound as a tuple to the `variables`
           # instance variable.
           self.variables = (x, y)

           # The forward pass should be performed using numpy arrays,
           # not the tensors themselves.
           x_arr = x.data
           y_arr = y.data
           return x_arr * y_arr

       def backward_var(self, grad, index, **kwargs):
           """Given ``grad = dℒ/df``, computes ``∂ℒ/∂x`` and ``∂ℒ/∂y``

           ``ℒ`` is assumed to be the terminal node from which ``ℒ.backward()`` was
           called.

           Parameters
           ----------
           grad : numpy.ndarray
               The back-propagated total derivative with respect to the present
               operation: dℒ/df. This will have the same shape as f, the result
               of the forward pass.

           index : Literal[0, 1]
               The index-location of ``var`` in ``self.variables``

           Returns
           -------
           numpy.ndarray
               ∂ℒ/∂x_{i}

           Raises
           ------
           SkipGradient"""
           x, y = self.variables
           x_arr = x.data
           y_arr = y.data

           if index == 0:  # backprop through a
               return grad * y.data  # ∂ℒ/∂x = (∂ℒ/∂f)(∂f/∂x)
           elif index == 1:  # backprop through b
               return grad * x.data  # ∂ℒ/∂y = (∂ℒ/∂f)(∂f/∂y)


   # Our function stitches together our operation class with the
   # operation arguments via `mygrad.prepare_op`
   def custom_multiply(x: ArrayLike, y: ArrayLike) -> mg.Tensor:
       # `prepare_op` will take care of casting `x` and `y` to tensors if
       # they are not already tensors.
       return prepare_op(CustomMultiply, x, y)

We can now use our differentiable function! It will automatically be compatible
with broadcasting; out operation need not account for broadcasting in either the
forward pass or the backward pass.

.. code:: pycon

   >> x = mg.tensor(2.0)
   >> y = mg.tensor([1.0, 2.0, 3.0])

   >> custom_multiply(x, y).backward()
   >> x.grad, y.grad
   (array(6.), array([2., 2., 2.]))

Documentation for mygrad.Operation
----------------------------------

.. currentmodule:: mygrad.operation_base

.. autosummary::
   :toctree: generated/

   Operation
   Operation.backward
   Operation.backward_var
   BroadcastableOp
