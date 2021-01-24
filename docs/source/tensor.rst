MyGrad's Tensor
***************
:class:`~mygrad.Tensor` is the most critical piece of MyGrad. It is a
numpy-array-like object capable of serving as a node in a computational
graph that supports back-propagation of derivatives via the chain rule.

You can effectively do a drop-in replacement of a numpy array with a :class:`~mygrad.Tensor`
for all basic mathematical operations. This includes `basic and advanced indexing <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html#Introducing-Basic-and-Advanced-Indexing>`_,
`broadcasting <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html>`_, sums `over axes <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/VectorizedOperations.html#Specifying-the-axis-Keyword-Argument-in-Sequential-NumPy-Functions>`_, etc; it will simply just work.

>>> import mygrad as mg  # note that we replace numpy with mygrad here
>>> x = mg.arange(9).reshape(3, 3)
>>> x
Tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
>>> y = x[x == 4] ** 2
>>> y
Tensor([16], dtype=int32)

Thus MyGrad users can spend their time mastering `numpy <https://www.pythonlikeyoumeanit.com/module_3.html>`_
and their skills will transfer seamlessly when using this autograd library.

Creating a Tensor
-----------------
:class:`~mygrad.Tensor` can be passed any "array-like" object of numerical data.
This includes numbers, sequences (e.g. lists), nested sequences, numpy-ndarrays,
and other mygrad-tensors. mygrad also provides familiar numpy-style tensor-creation
functions (e.g. :func:`~mygrad.arange`, :func:`~mygrad.linspace`, etc.)

>>> import mygrad as mg
>>> mg.Tensor(2.3)  # creating a 0-dimensional tensor
Tensor(2.3)
>>> mg.Tensor(np.array([1.2, 3.0]))  # casting a numpy-array to a tensor
Tensor([1.2, 3.0])
>>> mg.Tensor([[1, 2], [3, 4]])  # creating a 2-dimensional tensor from lists
Tensor([[1, 2],
       [3, 4]])
>>> mg.arange(4)    # using numpy-style tensor creation functions
Tensor([0, 1, 2, 3])

Integer-valued tensors are treated as constants

>>> mg.astensor(1, dtype=np.int8).constant
True

By default, float-valued tensors are not treated as constants

>>> mg.astensor(1, dtype=np.float32).constant
False

Forward and Back-Propagation
----------------------------
Let's construct a computational graph consisting of two zero-dimensional
tensors, ``x`` and ``y``, which are used to compute an output tensor,
``ℒ``. This is a "forward pass imperative" style for creating a computational
graph - the graph is constructed as we carry out the forward-pass computation.

>>> x = Tensor(3.0)
>>> y = Tensor(2.0)
>>> ℒ = 2 * x + y ** 2

Invoking ``ℒ.backward()`` signals the computational graph to
compute the total-derivative of ``f`` with respect to each one of its dependent
variables. I.e. ``x.grad`` will store ``dℒ/dx`` and ``y.grad`` will store
``dℒ/dy``. Thus we have back-propagated a gradient from ``f`` through our graph.

Each tensor of derivatives is computed elementwise. That is, if ``x = Tensor(x0, x1, x2)``,
then ``dℒ/dx`` represents ``[dℒ/d(x0), dℒ/d(x1), dℒ/d(x2)]``

>>> ℒ.backward()  # computes df/dx and df/dy
>>> x.grad  # df/dx
array(6.0)
>>> y.grad  # df/dy
array(4.0)
>>> ℒ.grad
array(1.0)  # dℒ/dℒ

Once the gradients are computed, the computational graph containing ``x``,
``y``, and ``ℒ`` is cleared automatically. Additionally, involving any
of these tensors in a new computational graph will automatically null
their gradients.

>>> 2 * x
>>> x.grad is None
True

Or, you can use the :func:`~mygrad.Tensor.null_grad` method to manually clear a
tensor's gradient

>>> y.null_grad()
Tensor(2.)
>>> y.grad is None
True


Accessing the Underlying NumPy Array
------------------------------------
:class:`~mygrad.Tensor` is a thin wrapper on ``numpy.ndarray``. A tensor's
underlying numpy-array can be accessed via ``.data``. This returns
a direct reference to the numpy array.

>>> x = mg.Tensor([1, 2])
>>> x.data
array([1, 2])

>>> import numpy as np
>>> np.asarray(x)
array([1, 2])

Producing a "View" of a Tensor
------------------------------
MyGrad's tensors exhibit the same view semantics and memory-sharing relationships
as NumPy arrays. I.e. any (non-scalar) tensor produced via basic indexing will share
memory with its parent.

>>> x = mg.Tensor([1., 2., 3., 4.])
>>> y = x[:2]  # the view: Tensor([1., 2.])
>>> y.base is x
True
>>> np.shares_memory(x, y)
True

Mutating shared data will propagate through views:

>>> y *= -1
>>> x
Tensor([-1., -2.,  3.,  4.])
>>> y
Tensor([-1., -2.])

And this view relationship will also manifest between the tensors' gradients

>>> (x ** 2).backward()
>>> x.grad
array([-2., -4.,  6.,  8.])
>>> y.grad
array([-2., -4.])

Documentation for mygrad.Tensor
-------------------------------

.. currentmodule:: mygrad

.. autosummary::
   :toctree: generated/

   Tensor
   Tensor.astype
   Tensor.backward
   Tensor.base
   Tensor.clear_graph
   Tensor.constant
   Tensor.copy
   Tensor.creator
   Tensor.dtype
   Tensor.grad
   Tensor.item
   Tensor.ndim
   Tensor.null_grad
   Tensor.null_gradients
   Tensor.shape
   Tensor.size
   Tensor.T



