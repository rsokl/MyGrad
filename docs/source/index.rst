.. MyGrad documentation master file, created by
   sphinx-quickstart on Sun Oct 21 09:57:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
MyGrad
======
MyGrad is a lightweight library that adds automatic differentiation to NumPy – its only 
dependency is NumPy. Simply "drop in" a MyGrad tensor into your NumPy-based code, and 
start differentiating!

.. code-block:: pycon

   >>> import mygrad as mg
   >>> import numpy as np

   >>> x = mg.tensor([1., 2., 3.])  # like numpy.array, but supports backprop
   >>> f = np.sum(x * x)  # tensors can be passed directly to native numpy functions!
   >>> f.backward() # triggers automatic differentiation
   >>> x.grad  # stores [df/dx0, df/dx1, df/dx2]
   array([2., 4., 6.])


MyGrad's primary goal is to make automatic differentiation accessible and easy to use across the Python/NumPy ecosystem.
As such, it strives to behave and feel exactly like NumPy so that users need not learn yet another array-based math library.

Of the various modes and flavors of auto-diff, MyGrad currently only supports back-propagation from a scalar quantity.


"Drop in" automatic differentiation?
====================================
What we mean by drop in automatic differentiation is that you can take a third party function, which is written in NumPy, and pass MyGrad tensors as its inputs – this will coerce it into using MyGrad functions internally so that we can differentiate the function.

.. code-block:: python
   :caption: What we mean by drop in autodiff

   from third_party_lib import some_numpy_func
   
   import mygrad as mg

   arr1 = mg.tensor(...) # some MyGrad Tensor (instead of a NumPy array)
   arr2 = mg.tensor(...) # some MyGrad Tensor (instead of a NumPy array)

   output = some_numpy_func(arr1, arr2)  # "drop in" the MyGrad tensors

   output.backward()  # output is a MyGrad tensor, not a NumPy array!

   arr1.grad  # stores d(some_numpy_func) / d(arr1)
   arr2.grad  # stores d(some_numpy_func) / d(arr2)


MyGrad aims for parity with NumPy's major features
==================================================
NumPy's ufuncs are richly supported. We can even differentiate through an operation that occur in-place on a tensor and applies a boolean mask to
the results:

.. code-block:: pycon

   >>> x = mg.tensor([1., 2., 3.])
   >>> y = mg.zeros_like(x)
   >>> np.multiply(x, x, where=[True, False, True], out=y)
   >>> y.backward()
   >>> x.grad
   array([2., 0., 6.])


NumPy's `view semantics <https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html#Producing-a-View-of-an-Array>`_ are also mirrored to a high fidelity: performing basic
indexing and similar operations on tensors will produce a "view" of that tensor's data, thus a tensor and its view share memory.
This relationship will also manifest between the derivatives stored by a tensor and its views!

.. code-block:: pycon

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

.. code-block:: pycon

   >>> (x[x < 4] ** 2).backward()
   >>> x.grad
   array([[0., 2., 4.],
          [6., 0., 0.],
          [0., 0., 0.]])


NumPy arrays and other array-likes play nicely with MyGrad's tensor. These behave like constants
during automatic differentiation

.. code-block:: pycon

   >>> x = mg.tensor([1., 2., 3.])
   >>> constant = [-1., 0., 10]  # can be a numpy array, list, or any other array-like
   >>> (x * constant).backward()  # all array-likes are treated as constants
   >>> x.grad
   array([-1.,  0., 10.])


What About JAX?
===============
Doesn't JAX already provide drop in automatic differentiation? Not quite; JAX provides *swap-out* automatic differentiation: you must swap out the version of NumPy you are using *before* you write your code. Thus you cannot simply differentiate some third party function by passing it a JAX array.

"Is MyGrad a competitor to JAX? Should I stop using JAX and start using MyGrad?"

**Goodness gracious, no!** MyGrad is *not* meant to compete with the likes of JAX, which offers far more functionality in the way of computing higher-order derivatives, Jacobian vector projects, in terms of providing a jit... this list goes on. 
MyGrad is meant to be a simple and highly accessible way to provide basic automatic differentiation capabilities to the NumPy ecosystem. Anyone who knows how to use NumPy can very easily learn to use MyGrad. It is especially great for teaching. But once your auto-diff needs extend beyond derivatives of scalars, it is time to graduate to JAX.


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
   io
   graph_viz
   changes
