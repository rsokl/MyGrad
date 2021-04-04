.. _performance-tips:

Performance Tips
****************

The following functions provide users with controls for optimizing MyGrad code
by either suspending its memory-guarding behavior or by disabling automatic differentiation
altogether. These are important utilities for speeding up your code.

Beyond the points made below, general performance tips for NumPy – e.g. leveraging
`vectorized operations <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/VectorizedOperations.html>`_,
heeding NumPy's `row-major memory layout for arrays <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/ArrayTraversal.html>`_
when constructing tensors, and using `basic indexing <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html>`_
to create views of arrays instead of copies – apply equally to MyGrad and its tensors.
After all, MyGrad operates almost entirely in NumPy arrays and NumPy functions under the hood.


.. currentmodule:: mygrad

Suspending Graph-Tracking for Automatic Differentiation
-------------------------------------------------------
.. autosummary::
   :toctree: generated/

   no_autodiff

In the case that you want to run a computation involving MyGrad tensors, but you don't need to access
their gradients (e.g. when measuring the "test-time" performance of a model that you are training), then
you can use the provided decorator/context-manager for suspending all of MyGrad's
"graph-tracking" features.

.. code-block:: python

   >>> import mygrad as mg
   >>> with mg.no_autodiff:
   ...     # any mygrad code in this context will run faster
   ...     # but will not produce any gradients


Note that this also suspends all memory-guarding (see below), since MyGrad doesn't need to ensure the
preservation of any state.

Suspending all graph-tracking features can speed up code involving many small tensors substantially - about
a 3x speedup.


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


Note that, were ``x`` an instance of :class:`~mygrad.Tensor`, there would not be any issue with the
above calculation, since MyGrad can track the in-place update on a tensor. MyGrad cannot, on the otherhand
track such operations involving only NumPy arrays

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
Note, however, for computations involving large tensors (e.g. for typical dense and convolutional neural networks), the
overhead associated with the memory-guarding feature is likely negligible compared to the core numerical computations
at play.

If one wants to enjoy the optimizations associated with removing memory guarding, it is recommended that you first test
your code with the default memory guarding enabled; once you have witnessed that MyGrad didn't raise any errors, you can
then proceed to run your code "at scale" with memory-guarding disabled.


Make Use of Views but Avoid Involving them in In-Place Operations
-----------------------------------------------------------------

Please refer to the section on views and in-place operations for more details.
The upshot is: views of tensors are efficient to create, as they do not involve copying any memory, but performing
an in-place operations on a tensor will copy that tensor. Furthermore, performing an in-place operation on a view
will lead to the creation of a copy of its associated base tensor.

If you are relying on this mutation propagating to many various views, then this can still be a net-gain in performance
compared to updating all of them "manually". But, generally, in-place updates on tensors do not have the same performance
benefits as do augmentations on NumPy arrays.