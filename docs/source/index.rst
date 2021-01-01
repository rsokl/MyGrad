.. MyGrad documentation master file, created by
   sphinx-quickstart on Sun Oct 21 09:57:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MyGrad
======
MyGrad is a simple, NumPy-centric math library that is capable of performing *automatic differentiation*. That is, the
mathematical functions provided by MyGrad are capable of computing their own derivatives. If you know `how to use NumPy
<https://www.pythonlikeyoumeanit.com/module_3.html>`_ then you can learn how to use MyGrad in a matter of minutes!

Let's use ``mygrad`` to compute the derivative of
:math:`f(x) = x^2` evaluated at :math:`x = 3` (which is :math:`\frac{df}{dx}\rvert_{x=3} = 2\times 3`).

:class:`~mygrad.Tensor` behaves nearly identically to NumPy's ndarray, in addition to having the machinery needed to
compute the analytic derivatives of functions. Suppose we want to compute this derivative at ``x = 3``. We can create a
0-dimensional tensor (a scalar) for x and compute ``f(x)``:

.. code:: pycon

    >>> import mygrad as mg
    >>> x = mg.Tensor(3.0)
    >>> f = x ** 2
    >>> f
    Tensor(9.0)


Invoking :meth:`~mygrad.Tensor.backward` on ``f`` instructs ``mygrad`` to trace through the computational graph that produced ``f`` and compute the
derivatives of ``f`` with respect to all of its independent variables. Thus, executing ``f.backward()`` will compute :math:`\frac{df}{dx} = 2x` at :math:`x=3`, and will store the resulting value in ``x.grad``:

.. code:: pycon

    >>> f.backward()  # triggers computation of ``df/dx``
    >>> x.grad  # df/dx = 2x = 6.0
    array(6.0)


While fantastic auto-differentiation libraries like TensorFlow, PyTorch, and JAX are available to the same end as
MyGrad (and far far beyond, ultimately), they are industrial-grade tools in both function and form. MyGrad's primary purpose
is to serve as an educational tool. It is simple to install (its only core dependency in NumPy), it is trivial to use
if you are comfortable with NumPy, and its code base is well-documented and easy to understand. This makes it simple for
students and teachers alike to use, hack, prototype with, and enhance MyGrad!

Why is Automatic Differentiation Useful?
----------------------------------------
In general, auto-differentiation permits us to compute massive equations that depend on millions of variables and then
seamlessly evaluate the derivatives of the equation's output *with respect to every one of those variables*. This
capability lies at the heart of the burgeoning field of **deep learning**, which is now the predominant use case for
auto-differentiation libraries, and is the manifest purpose of TensorFlow, PyTorch, and MXNet.

The "decisions" made by a neural network are dictated by the network's many, many parameters, which us researchers have
arranged to serve as variables in a tremendous equation. This equation might, for example, attempt to take as input the pixels
of a picture and return as an output an image-classification - a statement of the image's content (e.g. 0 is 'dog',
1 is 'cat', etc.).

The way that we train this neural network is by "tuning" the values of its many parameters so that the network's
predictions reliably agree with what we know to be true. It turns out that having access to the derivative of the
neural network's output with respect to its parameters grants us the ability to quite reliably optimize its parameters -
through a process known as gradient-based optimization we can update the values of these parameters to steer the neural
network towards making more faithful predictions (note: a gradient is just a collection of derivatives of a
multivariate function).

More specifically, we can hook our neural network up to an "objective" function that measures how well its predictions
match against "the truth". Recalling the basic definition of a derivative (as prescribed by any calculus course) and its
relationship to the slope of a function at a point, knowing
the derivative of this objective function with respect to one of our neural network's parameters means that we know whether
increasing this parameter will increase or decrease the output of the objective function; tuning the parameter so will affect
the network's output such that its prediction is in closer agreement with the truth than before. If we make such an
adjustment to each of our neural network's parameters and repeat this process many times over, using a wide variety of
"training data" we may arrive at a configuration of network parameters that permits our neural network to faithfully
classify pictures that we have never encountered before.

Thus auto-differentiation permits us to efficiently and automatically compute the derivatives of massive functions by
way of simply coding the functions using the auto-differentiation software. This in turn, is what allows us nimbly design
neural networks and objective functions, and to tune the parameters of our neural networks using derivative-based (or
gradient-based) optimization schemes.

It should be noted that description of training neural networks, as presented here, only provides a narrow view of deep learning.
Specifically, it describes the supervised learning of an image classification problem. While this is sufficient for conveying
the utility of auto-differentiation software as a means for training neural networks, there is more nuiance to deep learning
than is suggested here.



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
