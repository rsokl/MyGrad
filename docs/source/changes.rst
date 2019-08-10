=========
Changelog
=========

This is a record of all past mygrad releases and what went into them,
in reverse chronological order. All previous releases should still be available
on pip.

.. _v1.2.0:

-------------------
1.2.0 - 2019-08-03
-------------------

We're finally keeping a formal changelog!

This release makes substantial improvements to MyGrad's error-checking and handling, in order to make much simpler the process of debugging issues with buggy custom operations. Specifically, :func:`~mygrad.operation_base.Operation.backward` now checks for an invalid-gradients on each call of :func:`~mygrad.operation_base.Operation.backward_var`, and raises a descriptive error message.

``mygrad.errors`` was introduced to provide descriptive, mygrad-specific exceptions. For example, we no longer raise bare exceptions for scenarios like invalid backprop through a scalar-only graph, rather we now raise a descriptive ``InvalidBackprop`` exception.

:func:`~mygrad.maximum` and :func:`~mygrad.minimum` were patched to permit backpropagation through scalar inputs.

Internal implementation details of :func:`~mygrad.einsum` were adjusted to remove redundant code in the its backpropagation machinery.

MyGrad's testing framework was improved to achieve more complete test coverage and to eliminate some flaky testing behavior.

