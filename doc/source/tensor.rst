Tensor
******
A numpy-array-like object capable of serving as a node in a computational
graph that supports back-propagation of derivatives via the chain rule.
See the Examples section of the docstring for more details.

Like the numpy array, mygrad's tensor stores data as an N-dimensional array
and provides an interface accessing, setting, and performing vectorized
operations along the various dimensions of this array. Vectorized operations
support numpy-style broadcasting semantics.

The contents of a tensor can be accessed and written to using all variety
of basic and advanced indexing (along with mixtures of the two).

Creating a Tensor
-----------------
``mygrad.Tensor`` can be passed any "array-like" object of numerical data.
This includes numbers, sequences (e.g. lists), nested sequences, numpy-ndarrays,
and other mygrad-tensors. mygrad also provides familiar numpy-style tensor-creation
functions (e.g. ``mygrad.arange``, ``mygrad.linspace``, etc.)

>>> import mygrad as mg
>>> mg.Tensor(2.3)  # creating a 0-dimensional tensor
Tensor(2.3)
>>> mg.Tensor(np.array([1.2, 3.0]))  # casting a numpy-array to a tensor
Tensor([1.2, 3.0])
>>> mg.Tensor([[1, 2], [3, 4]])  # creating a 2-dimensional tensor
Tensor([[1, 2],
       [3, 4]])
>>> mg.arange(4)    # using numpy-style tensor creation functions
Tensor([0, 1, 2, 3])


Forward and Back-Propagation
----------------------------
Let's construct a computational graph consisting of two zero-dimensional
tensors, ``x`` and ``y``, which are used to compute an output tensor,
``f``. This is a "forward pass imperative" style for creating a computational
graph - the graph is constructed as we carry out the forward-pass computation.

>>> x = Tensor(3.0)
>>> y = Tensor(2.0)
>>> f = 2 * x + y ** 2

Invoking ``f.backward()`` signals the computational graph to
compute the total-derivative of ``f`` with respect to each one of its dependent
variables. I.e. ``x.grad`` will store ``df/dx`` and ``y.grad`` will store
``df/dy``. Thus we have back-propagated a gradient from ``f`` through our graph.

Each tensor of derivatives is computed elementwise. That is, if ``x = Tensor(x0, x1, x2)``,
then df/dx represents ``[df/d(x0), df/d(x1), df/d(x2)]``

>>> f.backward()  # computes df/dx and df/dy
>>> x.grad  # df/dx
array(6.0)
>>> y.grad  # df/dy
array(4.0)
>>> f.grad
array(1.0)  # df/df

Before utilizing ``x`` and ``y`` in a new computational graph, you must
'clear' their stored derivative values. ``f.null_gradients()`` signals
``f`` and all preceding tensors in its computational graph to clear their
derivatives.

>>> f.null_gradients()
>>> x.grad is None and y.grad is None and f.grad is Nonw
True

Accessing the Underlying NumPy Array
------------------------------------
``mygrad.Tensor`` is a thin wrapper on ``numpy.ndarray``. A tensor's
underlying numpy-array can be accessed via ``.data``:

>>> x = mg.Tensor([1, 2])
>>> x.data
array([1, 2])

**Do not modify this underlying array**. Any in-place modifications made to this
array will not be tracked by any computational graph involving that tensor, thus
back-propagation through that tensor will likely be incorrect.


.. currentmodule:: mygrad

.. autosummary::
   :toctree: generated/

   Tensor

