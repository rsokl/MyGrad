=========
Changelog
=========

This is a record of all past mygrad releases and what went into them,
in reverse chronological order. All previous releases should still be available
on pip.

.. _v1.4.0:

-------------------
1.4.0 - 2019-12-19
-------------------

This release adds the ``mygrad.repeat`` operation. It also includes some minor
improvements to mygrad's test suite.


.. _v1.3.0:

-------------------
1.3.0 - 2019-11-30
-------------------

This release adds ``mygrad.clip`` and ``mygrad.where``.

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

