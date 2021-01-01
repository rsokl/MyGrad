#############################
Views and In-Place Operations
#############################

Producing a "View" of a Tensor
==============================

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

In-Place Operations are not Efficient
=====================================
It is important to note that although MyGrad's view semantics promote a rich parity
with NumPy, certain aspects should be avoided in the interest of optimized performance.
Namely, performing in-place operations on tensors is generally not more efficient than
their non-mutating counterparts.

This is because MyGrad has to track the state of tensors that are involved in a computational
graph. Thus a mutated tensor must have its pre-augmented state stored for future reference; this
defeats the performance benefit of writing to an array's memory in-place. This is especially
inefficient if you are mutating a tensor involved with multiple views of the same memory(
By contrast, producing a view of a tensor *is* efficient as one would expect).

Thus these NumPy-like in-place semantics are supported by MyGrad not for the same performance
purposes, but instead to support convenient and familiar code-patterns and to enable one to
port NumPy code to MyGrad (or, in the future, inject MyGrad tensors into NumPy!!) and get
the exact same behavior.

A final note: MyGrad's in-place operations, when run under :func:`~mygrad.no_autodiff` mode,
do not incur the extra costs noted above, and thus your code will benefit from the performance
benefits of in-place operations.