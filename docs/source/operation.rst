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


Explaining Scalar-Only Operations
----------------------------------
MyGrad only supports gradients whose elements have a one-to-one correspondence
with the elements of their associated tensors. That is, if ``x`` is a shape-(4,)
tensor:

.. math::

   x = [x_0, x_1, x_2, x_3]

then the gradient, with respect to ``x``, of the terminal node of our computational graph (:math:`l`) must
be representable as a shape-(4,) tensor whose elements correspond to those of ``x``:

.. math::

  \nabla_{x}{l} = [\frac{dl}{dx_0}, \frac{dl}{dx_1}, \frac{dl}{dx_2}, \frac{dl}{dx_3}]

If an operation class has ``scalar_only=True``, then the terminal node of a
computational graph involving that operation can only trigger back-propagation
from a 0-dimensional tensor (i.e. a scalar). This is ``False`` for operations that
manifest as trivial element-wise operations over tensors. In such cases, the
gradient of the operation can also be treated element-wise, and thus be computed
unambiguously.

The matrix-multiplication operation, for example, is a scalar-only operation because
computing the derivative of :math:`F_{ik} = \sum_{j}{A_{ij} B_{jk}}` with respect
to each element of :math:`A` produces a 3-tensor: :math:`\frac{d F_{ik}}{d A_{iq}}`, since each element 
of :math:`F` depends on *every* element in the corresponding row of :math:`A`.
This is the case unless the terminal node of this graph is eventually reduced (via summation, for instance) to a
scalar, :math:`l`, in which  case the elements of the 2-tensor :math:`\frac{dl}{dA_{pq}}` has a trivial one-to-one
correspondence to the elements of :math:`A_{pq}`.


Documentation for mygrad.Operation
----------------------------------

.. currentmodule:: mygrad.operation_base

.. autosummary::
   :toctree: generated/

   Operation
   Operation.backward
   Operation.backward_var
   BroadcastableOp
