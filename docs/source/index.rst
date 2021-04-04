.. MyGrad documentation master file, created by
   sphinx-quickstart on Sun Oct 21 09:57:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MyGrad
======
MyGrad is a lightweight library that adds automatic differentiation to NumPy – its only dependency is NumPy!

.. code:: python

   >>> import mygrad as mg
   >>> import numpy as np

   >>> x = mg.tensor([1., 2., 3.])  # like numpy.array, but supports backprop!
   >>> f = np.sum(x * x)  # tensors work with numpy functions!
   >>> f.backward() # triggers automatic differentiation
   >>> x.grad  # stores [df/dx0, df/dx1, df/dx2]
   array([2., 4., 6.])


MyGrad's primary goal is to make automatic differentiation an accessible and easy to use across the Python/NumPy ecosystem.
As such, it strives to behave and feel exactly like NumPy so that users need not learn yet another array-based math library.

NumPy's ufuncs are richly supported. We can even differentiate through an operation that occur in-place on a tensor and applies a boolean mask to
the results:

.. code:: python

   >>> x = mg.tensor([1., 2., 3.])
   >>> y = mg.zeros_like(x)
   >>> np.multiply(x, x, where=[True, False, True], out=y)
   >>> y.backward()
   >>> x.grad
   array([2., 0., 6.])


NumPy's `view semantics <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html#Producing-a-View-of-an-Array>`_ are also mirrored to a high fidelity: performing basic
indexing and similar operations on tensors will produce a "view" of that tensor's data, thus a tensor and its view share memory.
This relationship will also manifest between the derivatives stored by a tensor and its views!

.. code:: python

   >>> x = mg.arange(9.).reshape(3, 3)
   >>> diag_view = np.einsum("ii->i", x)  # returns a view of the diagonal elements of `x`
   >>> x, diag_view
   (Tensor([[0., 1., 2.],
   [3., 4., 5.],
   [6., 7., 8.]]),
   Tensor([0., 4., 8.]))

   # views share memory
   >>> np.shares_memory(x, diag_view)
   True

   # mutating a view affects its base (and all other views)
   >>> diag_view *= -1  # mutates x in-place
   >>> x
   Tensor([[-0.,  1.,  2.],
           [ 3., -4.,  5.],
           [ 6.,  7., -8.]])

   >>> (x ** 2).backward()
   >>> x.grad, diag_view.grad
   (array([[ -0.,   2.,   4.],
           [  6.,  -8.,  10.],
           [ 12.,  14., -16.]]),
    array([ -0.,  -8., -16.]))

   # the gradients have the same view relationship!
   >>> np.shares_memory(x.grad, diag_view.grad)
   True


Basic and advanced indexing is fully supported

.. code:: python

   >>> (x[x < 4] ** 2).backward()
   >>> x.grad
   array([[0., 2., 4.],
          [6., 0., 0.],
          [0., 0., 0.]])


NumPy arrays and other array-likes play nicely with MyGrad's tensor. These behave like constants
during automatic differentiation

.. code:: python

   >>> x = mg.tensor([1., 2., 3.])
   >>> constant = [-1., 0., 10]  # can be a numpy array, list, or any other array-like
   >>> (x * constant).backward()  # all array-likes are treated as constants
   >>> x.grad
   array([-1.,  0., 10.])




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   intro
   tensor
   views
   performance_tips
   operation
   tensor_creation
   tensor_manipulation
   linalg
   math
   indexing
   nnet
   graph_viz
   changes
