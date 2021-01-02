=========
Changelog
=========

This is a record of all past mygrad releases and what went into them,
in reverse chronological order. All previous releases should still be available
on pip.

.. _v2.0.0:

------------------
2.0.0 - 2021-01-01
------------------

🎉🎉🎉

This is a compatibility-breaking update to MyGrad, and it's great!
MyGrad 2.0 represents a major overhaul to this project.
Its primary feature is that MyGrad now offers the ability to create and augment views of
tensors.
This enables a large variety of convenient code patterns that were impossible in MyGrad 1.X.
It also creates near parity between the experiences of using MyGrad and using NumPy
(which, in turn, paves the way for injecting autodiff functionality *into* NumPy code via MyGrad in
a future release!).

Another important, but less exciting, feature is that MyGrad now protects users from inadvertently
corrupting the state of a computational graph by, say, mutating a NumPy array that is participating in
the graph.
This is very important to protect people – especially students – from unwittingly poisoning the results
of their calculations.

Lastly... no more "nulling" gradients! MyGrad will now handle deleting gradients for you in a way that
is nicely compatible with gradient-based optimization work flows.

New Utilities
-------------

- :func:`~mygrad.astensor`
- :func:`~mygrad.asarray`
- :func:`~mygrad.no_autodiff`
- :func:`~mygrad.mem_guard_off`
- :func:`~mygrad.mem_guard_on`
- :func:`~mygrad.turn_memory_guarding_off`
- :func:`~mygrad.turn_memory_guarding_on`



Augmented Updates on Tensors Now Match NumPy's Behavior
-------------------------------------------------------

Previously, augmented assignment expressions, such as ``tensor *= 2``, behaved merely
as a shorthand for the simple assignment ``tensor = tensor * 2``.
This is in stark contrast to the behavior of an augmented assignment on a NumPy array, which
`mutates the array in-place <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html#Augmented-Assignments>`_.

This meant that there was a major discrepancy between how these expressions behaved across MyGrad and
NumPy.
This has changed in MyGrad 2.0: all augmented assignment expressions operate in-place on tensors and
mutate their underlying data.

+-----------------------------------+-----------------------------------+-----------------------------------+
| Numpy                             | MyGrad 1.X                        | MyGrad 2.0                        |
+===================================+===================================+===================================+
| .. code:: python                  | .. code:: python                  | .. code:: python                  |
|                                   |                                   |                                   |
|    >>> x = np.array([1., 2.])     |    >>> x = mg.Tensor([1., 2.])    |    >>> x = mg.Tensor([1., 2.])    |
|    >>> y = x                      |    >>> y = x                      |    >>> y = x                      |
|    >>> x *= 2                     |    >>> x *= 2                     |    >>> x *= 2                     |
|    >>> x is y                     |    >>> x is y  # doesn't match!   |    >>> x is y  # matches!         |
|    True                           |    False                          |    True                           |
+-----------------------------------+-----------------------------------+-----------------------------------+



Creating and Augmenting Views of Tensors
----------------------------------------

MyGrad now provides rich support for creating and manipulating views of tensors.

All `basic indexing <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html#>`_ operations
performed on a tensor will produce a view of said tensor.
This means that these two tensors share memory
(While MyGrad 1.X created a view of the underlying NumPy array under the hood for basic indexing, its notion
of supporting views went no further than that.)
As with NumPy arrays the "parent" of a view can be accessed through the tensor's ``.base``
attribute

+-----------------------------------+-------------------------------------+-----------------------------------+
| Numpy                             | MyGrad 1.X                          | MyGrad 2.0                        |
+===================================+=====================================+===================================+
| .. code:: python                  | .. code:: python                    | .. code:: python                  |
|                                   |                                     |                                   |
|    >>> x = np.array([1., 2., 3.]) |    >>> x = mg.Tensor([1., 2., 3.])  |    >>> x = mg.Tensor([1., 2., 3.])|
|    >>> y = x[:2]  # the view      |    >>> y = x[:2]  # the view        |    >>> y = x[:2]  # the view      |
|    >>> np.shares_memory(x, y)     |    >>> np.shares_memory(x, y)       |    >>> np.shares_memory(x, y)     |
|    True                           |    True                             |    True                           |
|    >>> y.base is x                |    >>> y.base is x  # doesn't match!|    >>> y.base is x  # matches!    |
|    True                           |    <ERROR>                          |    True                           |
+-----------------------------------+-------------------------------------+-----------------------------------+


Mutating shared data will propagate through views:


+-----------------------------------+-------------------------------------+------------------------------------+
| Numpy                             | MyGrad 1.X                          | MyGrad 2.0                         |
+===================================+=====================================+====================================+
| .. code:: python                  | .. code:: python                    | .. code:: python                   |
|                                   |                                     |                                    |
|    >>> y *= -1                    |    >>> y *= -1                      |    >>> y *= -1                     |
|    >>> y                          |    >>> y                            |    >>> y                           |
|    array([-1., -2.])              |    Tensor([-1., -2.])               |    Tensor([-1., -2.])              |
|    >>> x                          |    >>> x  # doesn't match!          |    >>> x  # matches!               |
|    array([-1., -2., 3.])          |    Tensor([1., 2., 3.])             |    Tensor([-1., -2., 3.])          |
+-----------------------------------+-------------------------------------+------------------------------------+


Furthermore, views of tensors now propagate corresponding gradient information as well!
This means that if ``y`` is a view of ``x``, then ``y.grad`` will be a corresponding view of ``x.grad``.
This is true for all varieties of views, views of views, etc., of ``x``.

.. code-block:: python

   # Because `y` is a view of `x`, `y.grad` will be
   # a corresponding view of `x.grad`
   >>> (x ** 2).backward()
   >>> x.grad
   array([-2., -4.,  6.,  8.])
   >>> y.grad
   array([-2., -4.])
   >>> y.grad.base is x.grad
   True

This rich support for views, augmented assignments, and in-place updates on tensors enables much more sophisticated
operations on tensors now.
For example, let's make a shape-(3, 3) tensor and perform and operations involving views of its diagonal and
its anti-diagonal. (Note that :func:`~mygrad.einsum` is capable of returning a view of a tensor's diagonal,
and that  MyGrad fully supports backpropagation through all flavors of einsum!)

.. code-block:: python

   >>> x = mg.Tensor([[0., 1., 2.],
   ...                [3., 4., 5.],
   ...                [6., 7., 8.]])

   # view of diagonal of `x`
   >>> diag = mg.einsum("ii->i", x)
   >>> diag
   Tensor([0., 4., 8.])

   # view of anti-diagonal of `x`
   >>> anti_diag = mg.einsum("ii->i", x[:, ::-1])
   >>> anti_diag
   Tensor([2., 4., 6.])

   # Compute derivatives of their summed difference
   >>> (diag - anti_diag).sum().backward()
   >>> x.grad
   array([[ 1.,  0., -1.],
          [ 0.,  0.,  0.],
          [-1.,  0.,  1.]])

   # The views of `x` have the appropriate corresponding
   # views of `x.grad`
   >>> diag.grad
   array([1., 0., 1.])
   >>> anti_diag.grad
   array([-1.,  0., -1.])


Bye-Bye Null Gradients!
-----------------------

Gone are the days of having to manually clear your tensors' gradients and the computational graph that they were
in; now MyGrad does it for you!
This means that ``Tensor.null_gradients()`` no longer does anything other than emit a deprecation warning.
In an upcoming minor release this method will be removed entirely.

In MyGrad 2.0, calling :func:`~mygrad.Tensor.backward` will finish its computation by clearing the computational graph that was involved
in the backpropagation.
Thus any internally-referenced tensors associated with that computational graph become free for garbage collection.
This is very nice behavior to help prevent students from filling up their RAM unwittingly.

And instead of worrying about nulling gradients manually, a tensor will automatically have its gradient cleared any time that it is
involved in a new mathematical operation.
This enables the following common workflow for performing gradient-based optimization:

.. code-block:: python

   for _ in range(num_optimization_steps):
       # using `model_params` in a function will automatically
       # set its gradients to `None`
       loss = compute_loss(data, model_params)
       loss.backward()         # compute gradients
       optimize(model_params)  # do stuff with gradients


You can also call :func:`~mygrad.Tensor.null_grad` to manually clear an individual tensor's gradient.



Safety First: Memory Guarding Behavior in MyGrad 2.0
----------------------------------------------------

In MyGrad 1.X it was all too easy to unwittingly corrupt the state of a computational graph by mutating
a NumPy array mid-computation.
This could lead to incorrect calculations of gradients! This is the stuff of horrifying nightmares.

Now MyGrad tracks all of the arrays that are involved in active computational graphs and locks their memory
so that they are read-only (except for when the user mutates the array explicitly with a MyGrad operation).
This means that the sort of mutation that could have lurked silently in the dimly-lit alleyways of bugs-ville will
now get loudly narc'd on by MyGrad's merciless memory guard!


+---------------------------------------------+---------------------------------------+
| MyGrad 1.X                                  | MyGrad 2.0                            |
+=============================================+=======================================+
| .. code:: python                            | .. code:: python                      |
|                                             |                                       |
|    >>> arr = np.array([1., 2.])             |    >>> arr = np.array([1., 2.])       |
|    >>> tn = mg.Tensor([1. 1.])              |    >>> tn = mg.Tensor([1. 1.])        |
|    >>> z = x * y                            |    >>> z = x * y                      |
|    # mutating x will corrupt                |    # mutating x will corrupt          |
|    # backprop through z...                  |    # backprop through z...            |
|    >>> x[:] = 0.                            |    >>> x[:] = 0. # you shall not pass!|
|                                             |    ValueError: read-only!             |
|    >>> z.backward() # uh oh...              |    >>> z.backward()                   |
|    >>> tn.grad # should be: (1., 2.)        |    >>> tn.grad                        |
|    array([0., 0.])                          |    array([1., 2.])                    |
+---------------------------------------------+---------------------------------------+

Any tensor or array that is no longer participating in an active computational graph will automatically
have its write-ability restored to its original state.

.. code-block:: python

   # memory guarding is released once an array is no
   # longer involved in an active computational graph
   >>> import mygrad as mg
   >>> import numpy as np
   >>> x = np.array([1., 2.])
   >>> y = mg.ones_like(x)
   >>> z = x * y     # x and y are locked
   >>> z.backward()  # graph cleared; x and y are "released"
   >>> x[:] = 0      # can write to x
   >>> x
   array([0., 0.])

   >>> x * y  # result is not referenced, thus x and y are immediately released

But with great responsibility comes great ...uhh... slowness? This memory-guarding feature can lead to slowdowns
of **up to 50% for computations involving many small tensors**.
(It used to be **a lot** worse... like 5x worse. I worked really hard to speed it up! I promise!)
That being said, computations involving beefy tensors (e.g. standard neural networks) will not be significantly
affected by the overhead associated with the memory guard.
That being said, please refer to :ref:`performance-tips` for ways to disable this memory-guarding mechanism.

Speaking of optimizations...


Disabling Automatic Differentiation
-----------------------------------

Sometimes you want to use your MyGrad code to do calculations, but you don't actually need to compute
any derivatives.
A common example of this is evaluating the test-time performance of a machine learning model that you are
in the process of optimizing – you don't actually need to perform backpropagation when you are processing
the test data.

In these circumstances, you can greatly reduce the overhead cost associated with building a computational
graph by using the :func:`~mygrad.no_autodiff` decorator / context manager. See the linked documentation
for example usage.

For computations involving many small tensors, this can produce **up to a 3x speedup**! So make sure you
make keen use of this when you don't actually need to perform autodiff.


.. _v1.9.0:

------------------
1.9.0 - 2020-08-28
------------------

The most significant aspect of this release is the implementation of ``Tensor.__array__``, which enables a huge amount
of cross-compatibility with numpy utilities (`#288 <https://github.com/rsokl/MyGrad/pull/288>`_). Note that any previous
reliance of a numpy function to produce an array of tensor-scalars will likely produce a standard numpy array instead.

Improvements:

- ``x**1`` and ``x**2`` are now special-cased in order to make these common operations more efficient (`#266 <https://github.com/rsokl/MyGrad/pull/266>`_)
- The derivative of :func:`~mygrad.nnet.losses.focal_loss` was refactored to handle special edge-cases and the tests for focal loss were improved to exercise these edge cases (`#269 <https://github.com/rsokl/MyGrad/pull/269>`_)
- Various improvements to the tests (`#271 <https://github.com/rsokl/MyGrad/pull/271>`_, `#277 <https://github.com/rsokl/MyGrad/pull/277>`_, `#290 <https://github.com/rsokl/MyGrad/pull/290>`_, `#284 <https://github.com/rsokl/MyGrad/pull/284>`_, `#289 <https://github.com/rsokl/MyGrad/pull/289>`_, `#282 <https://github.com/rsokl/MyGrad/pull/282>`_, `#292 <https://github.com/rsokl/MyGrad/pull/292>`_, `#293 <https://github.com/rsokl/MyGrad/pull/293>`_)
- The internal mechanism for tracking tensors in computational graph now depends on hashing tensor-IDs instead of hashing tensors directly. The fact that tensors could be hashed was due to the fact that its equality specialty methods were being monkey-patched (`#276 <https://github.com/rsokl/MyGrad/pull/276>`_)
- :func:`~mygrad.nnet.activations.softmax` and :func:`~mygrad.nnet.activations.logsoftmax` both expose ``axis`` arguments (`#268 <https://github.com/rsokl/MyGrad/pull/268>`_)

Bug fixes:

-  `0D tensors could not be indexed into <https://github.com/rsokl/MyGrad/issues/272>`_ – e.g. to insert a newaxis (`#273 <https://github.com/rsokl/MyGrad/pull/273>`_)
- There was a potential numerical instability in :func:`mygrad.nnet.layers.batchnorm` (`#285 <https://github.com/rsokl/MyGrad/pull/285>`_)
- The ``dtype`` argument in ``Tensor.__init__`` was ignored when the array-like argument, x, was another Tensor-instance (`#294 <https://github.com/rsokl/MyGrad/pull/294>`_)

New features:

- ``Tensor.__array__`` now exposes the tensor's underlying numpy array – this enables a huge amount of cross-compatibility with numpy utilities (`#288 <https://github.com/rsokl/MyGrad/pull/288>`_)
- Adds :func:`~mygrad.asarray` (`#279 <https://github.com/rsokl/MyGrad/pull/279>`_)
- Adds :func:`~mygrad.astensor` (`#294 <https://github.com/rsokl/MyGrad/pull/294>`_)


.. _v1.8.1:

------------------
1.8.1 - 2020-07-28
------------------

This is an `internal change <https://github.com/rsokl/MyGrad/pull/265>`_ to the backprop
mechanism for ``Tensor.__getitem__``, which produces considerable speedups (2x-4x) for backprop
through basic indexing and boolean indexing. Thanks to Petar Griggs for finding this.


.. _v1.8.0:

------------------
1.8.0 - 2020-07-25
------------------

New features:

- Adds :func:`~mygrad.any` and :func:`~mygrad.Tensor.any`
- Adds :func:`~mygrad.random.rand`
- Adds :func:`~mygrad.random.randint`
- Adds :func:`~mygrad.random.randn`
- Adds :func:`~mygrad.random.random`
- Adds :func:`~mygrad.random.random_integers`
- Adds :func:`~mygrad.random.random_sample`
- Adds :func:`~mygrad.random.ranf`
- Adds :func:`~mygrad.random.sample`
- Adds :func:`~mygrad.random.seed`

Thanks to Darshan Krishnaswamy and Sam Carpenter for adding this functionality!

Fixes a bug in the GRU layer where mixed floating point precision dtypes between data and weights raised an error.
Thanks to Petar Griggs for the fix!

.. _v1.7.1:

------------------
1.7.1 - 2020-07-11
------------------

Fixes a bug in :func:`~mygrad.nnet.losses.negative_log_likelihood`, where setting ``constant=True`` had no effect.


.. _v1.7.0:

------------------
1.7.0 - 2020-07-11
------------------

This release continues the process of integrating functions from `mynn <https://github.com/davidmascharka/MyNN>`_.

New features:

- Adds :func:`~mygrad.nnet.initializers.glorot_normal`
- Adds :func:`~mygrad.nnet.initializers.glorot_uniform`
- Adds :func:`~mygrad.nnet.initializers.he_normal`
- Adds :func:`~mygrad.nnet.initializers.he_uniform`
- Adds :func:`~mygrad.nnet.initializers.normal`
- Adds :func:`~mygrad.nnet.initializers.uniform`
- Adds :func:`~mygrad.nnet.losses.focal_loss`
- Adds :func:`~mygrad.nnet.losses.negative_log_likelihood`

Big thanks to David Mascharka!

Improvements:

The interfaces to :func:`~mygrad.reshape` and :func:`~mygrad.Tensor.reshape` were adjusted to match exactly the interfaces to their NumPy counterparts.
I.e. :func:`~mygrad.reshape` now requires ``newshape`` to be a sequence, whereas :func:`~mygrad.Tensor.reshape` can accept an unpacked sequence for its
``newshape``.

:func:`~mygrad.Tensor.shape` is now settable - triggering an in-place reshape of a tensor, matching the corresponding behavior in NumPy.

Internal changes:

The logic for writing an in-place operation has been consolidated into a convenient wrapper: :func:`~mygrad.Tensor._in_place_op`.


.. _v1.6.0:

------------------
1.6.0 - 2020-06-21
------------------

New features:

- Adds :func:`~mygrad.nnet.activations.elu`
- Adds :func:`~mygrad.nnet.activations.glu`
- Adds :func:`~mygrad.nnet.activations.leaky_relu`
- Adds :func:`~mygrad.nnet.activations.selu`
- Adds :func:`~mygrad.nnet.activations.soft_sign`

Big thanks to David Mascharka!


.. _v1.5.0:

-------------------
1.5.0 - 2020-02-16
-------------------

New features:

- Adds :func:`~mygrad.Tensor.astype` method.
- Adds :func:`~mygrad.nnet.activations.hard_tanh`
- ``y_true`` can now be passed as a ``Tensor`` to :func:`~mygrad.nnet.losses.softmax_crossentropy`


This update also includes various improvements to the library's test suite.

.. _v1.4.1:

-------------------
1.4.1 - 2020-01-09
-------------------

This release performs an internal refactor in the ``nnet`` module of the library, as well as
an analogous refactor in the test suite. This also fixes a docstring in the ``multiclass_hinge``
loss to properly show a description in the readthedocs page.

.. _v1.4.0:

-------------------
1.4.0 - 2019-12-19
-------------------

This release adds the :func:`~mygrad.repeat` operation. It also includes some minor
improvements to mygrad's test suite.


.. _v1.3.0:

-------------------
1.3.0 - 2019-11-30
-------------------

This release adds :func:`~mygrad.clip` and :func:`~mygrad.where`.

It also includes a major fix to the graph-traversal mechanism for null-gradients and clear-graph,
eliminating an exponentially-scaling runtime.

``+x`` will now invoke ``mygrad.positive``, mirroring the numpy behavior

There are improvements to user-facing error messages and input validation in addition to major
improvements to mygrad's test suite. There is now a 100% line-coverage gate in mygrad's CI system.


.. _v1.2.0:

-------------------
1.2.0 - 2019-08-03
-------------------

We're finally keeping a formal changelog!

This release makes substantial improvements to MyGrad's error-checking and handling, in order to make much simpler the process of debugging issues with buggy custom operations. Specifically, :func:`~mygrad.operation_base.Operation.backward` now checks for an invalid-gradients on each call of :func:`~mygrad.operation_base.Operation.backward_var`, and raises a descriptive error message.

``mygrad.errors`` was introduced to provide descriptive, MyGrad-specific exceptions. For example, we no longer raise bare exceptions for scenarios like invalid backprop through a scalar-only graph; rather, we now raise a descriptive ``InvalidBackprop`` exception.

MyGrad's testing framework received wide-ranging improvements, yielding complete test coverage and fewer flaky tests. Coverage checks were added to the project's CI process.

:func:`~mygrad.maximum` and :func:`~mygrad.minimum` were patched to permit backpropagation through scalar inputs.

Internal implementation details of :func:`~mygrad.einsum` were adjusted to remove redundant code in its backpropagation machinery.

:func:`~mygrad.Tensor.null_gradients` was refactored to ensure that only a single traversal of the computational graph is performed to null all of the tensors' gradients. Furthermore, `Tensor.null_gradients(clear_graph=True)` now only performs a single graph traversal, instead of two.

In keeping with NumPy's behavior, performing `+x` (where `x` is a mygrad-tensor) no longer returns a reference of `x`, but returns `mygrad.positive(x)`.

Backpropagation through :func:`~mygrad.max` and :func:`~mygrad.min` now works for 0D tensors.

Input validation was added to :func:`mygrad.nnet.layers.utils.sliding_window_view`.

Fixed backpropagation through basic indexing, `x[ind] = b`, in which broadcasting occurred and `b` possess "excess" leading singleton dimensions.

