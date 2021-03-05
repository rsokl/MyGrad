MyGrad's Operation Class
************************
Base class for all tensor operations that support back-propagation
of gradients.

Consider the Operation-instance ``f``. A forward-pass through ``f`` is defined
via ``f.__call__``. Thus, given tensors ``a`` and ``b``, a computational
graph is defined ``f.__call__(a, b) -> c``, where the "creator" of tensor ``c``
is recorded as ``f``.::

       (tensor: a) --+
                      |-> [operation: f(a, b)] --> (tensor: c)
       (tensor: b) --+

Thus back-propagating through ``c`` will instruct ``f`` to back-propagate
the gradient to its inputs, which are recorded as ``a`` and ``b``. Each
node then back-propagates to any Operation-instance that is recorded
as its creator, and so on.


Documentation for mygrad.Operation
----------------------------------

.. currentmodule:: mygrad.operation_base

.. autosummary::
   :toctree: generated/

   Operation
   Operation.backward
   Operation.backward_var
   BroadcastableOp
