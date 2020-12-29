.. _routines.autodiff_utils:

Auto-Differentiation Utilities
******************************

The following functions provide users with controls for optimizing MyGrad code
by either suspending its memory-guarding behavior or by disabling automatic differentiation
altogether. The latter is valuable if you want to execute code involving MyGrad tensors,
but you don't need access to any gradients (e.g. if you are measuring test-time performance
of a model that you are training).


.. currentmodule:: mygrad

Controlling Memory-Guarding Behavior
------------------------------------
.. autosummary::
   :toctree: generated/

   mem_guard_off
   mem_guard_on
   turn_memory_guarding_off
   turn_memory_guarding_off


By default, MyGrad tracks and locks the readability of all of the NumPy arrays that are involved in computational graphs
involving tensors.

These stateful graphs are how MyGrad is able to perform backpropagation and
compute the gradients of tensors involved in a given calculation.
Because of the stateful nature of a computational graph, mutating a NumPy array inplace could
corrupt the state of the computational graph - i.e. the derivatives computed would not accurately
reflect the values that were used during the "forward pass".
Read the following code to see such a mutation rear its head.

.. code-block:: python

   >>> import mygrad as mg
   >>> import numpy as np
   >>> mg.turn_memory_guarding_off()  # speeds up calculations, but with risks involved..
   >>> x = np.arange(3.)
   >>> y = mg.ones_like(x)
   >>> z = x * y
   >>> x[:] = 0  # mutates x, corrupting state associated with z
   >>> z.backward()
   >>> y.grad  # would be array([0., 1., 2.]) if graph wasn't corrupted
   array([0., 0., 0.])

Thus MyGrad prohibits such mutations with its aforementioned "memory guarding" behavior, however it is
smart about restoring the writeability of all arrays once they are no longer participating in a computational
graph (e.g. backpropagation has been performed through the graph).

.. code-block:: python

   >>> import mygrad as mg
   >>> import numpy as np
   >>> x = np.arange(3.)
   >>> y = mg.ones_like(x)
   >>> z = x * y
   >>> try:
   ...     x[:] = 0  # raises because `x` is made read-only
   ... except ValueError:
   ...     pass
   >>> z.backward()
   >>> y.grad  # correct gradient is computed
   array([0., 1., 2.])
   >>> x[:] = 0  # the writeability of `x` is restored once backprop is complete

This memory-guarding behavior comes at a cost: for computations involving many small tensors (e.g. in an handmade RNN)
this can lead to slowdowns of ~50%. Thus MyGrad provides various mechanisms for disabling all such memory-guards.

If one wants to enjoy the optimizations associated with removing memory guarding, it is recommended that you first test
your code with the default memory guarding enabled; once you have witnessed that MyGrad didn't raise any errors, you can
then proceed to run your code "at scale" with memory-guarding disabled.



Suspending Graph-Tracking for Automatic Differentiation
-------------------------------------------------------
.. autosummary::
   :toctree: generated/

   no_autodiff

In the case that you want to run a computation involving MyGrad tensors, but you don't need to access
their gradients, then you can use the provided decorator/context-manager for suspending all of MyGrad's
"graph-tracking" features. Note that this also suspends all memory-guarding, since MyGrad doesn't need to
ensure the preservation of any state.

Suspending all graph-tracking features can speed up code involving many small tensors substantially - about
a 3x speedup.



