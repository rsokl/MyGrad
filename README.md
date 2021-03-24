[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![codecov](https://codecov.io/gh/rsokl/MyGrad/branch/master/graph/badge.svg)](https://codecov.io/gh/rsokl/MyGrad)
[![Documentation Status](https://readthedocs.org/projects/mygrad/badge/?version=latest)](https://mygrad.readthedocs.io/en/latest/?badge=latest)
[![Automated tests status](https://github.com/rsokl/MyGrad/workflows/Tests/badge.svg)](https://github.com/rsokl/MyGrad/actions?query=workflow%3ATests+branch%3Amaster)
[![PyPi version](https://img.shields.io/pypi/v/mygrad.svg)](https://pypi.python.org/pypi/mygrad)
![Python version support](https://img.shields.io/badge/python-3.7%20&#8208;%203.9-blue.svg)

# [MyGrad's Documentation](https://mygrad.readthedocs.io/en/latest/)

Please note that MyGrad 2.0 is currently available as a pre-release;
you can give it a whirl by installing it via the following command:

```shell script
pip install --pre mygrad
```

This represents a [massive overhaul of the library](https://mygrad.readthedocs.io/en/latest/changes.html), which adds lots of exciting features, like drop-in autodiff for
NumPy functions!

# Introducing mygrad
MyGrad is a lightweight (its only dependency is NumPy) autodiff library, that brings drop-in autodiff to NumPy!

```python
>>> import mygrad as mg
>>> import numpy as np

>>> x = mg.tensor([1., 2., 3.])  # like numpy.array, but supports backprop!
>>> np.sum(x * x).backward()  # works natively with numpy functions!
>>> x.grad
array([2., 4., 6.])
```

MyGrad's main goal is to make automatic differentiation accessible and easy to use for students and researchers alike!
As such, it strives to behave and feel exactly like NumPy so that users need not learn yet another array-based math library.  

NumPy's ufuncs are richly supported; e.g. we can autodiff through in-place targets and boolean masks:  

```python
>>> x = mg.tensor([1., 2., 3.])
>>> y = mg.zeros_like(x)
>>> np.multiply(x, x, where=[True, False, True], out=y)
>>> y.backward()
>>> x.grad
array([2., 0., 6.])
```

NumPy's view semantics are also mirrored to a high fidelity

```python
>>> x = mg.arange(9.).reshape(3, 3)
>>> diag_view = np.einsum("ii->i", x)
>>> x, diag_view
(Tensor([[0., 1., 2.],
         [3., 4., 5.],
         [6., 7., 8.]]),
 Tensor([0., 4., 8.]))

>>> np.shares_memory(x, diag_view)
True

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

>>> np.shares_memory(x.grad, diag_view.grad)
True
```

Basic and advanced indexing is fully supported

```python
>>> (x[x < 4] ** 2).backward()
>>> x.grad
array([[0., 2., 4.],
       [6., 0., 0.],
       [0., 0., 0.]])
```

NumPy arrays and other array-likes play nicely with MyGrad's tensor. These behave like constants
during automatic differentiation

```python
>>> x = mg.tensor([1., 2., 3.])
>>> y = np.array([-1., 0., 10])
>>> (x * y).backward()  # y is treated as a constant
>>> x.grad
array([-1.,  0., 10.])
```

## Installing mygrad (this project)
To install MyGrad, you can pip-install it:

```shell
pip install mygrad
```

or clone this repository and navigate to the MyGrad directory, then run: 
```shell
python setup.py install
```

MyGrad requires `numpy`. It is highly recommended that you utilize a release of `numpy` built with [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library)
 for access to optimized math routines.

## A Simple Application
Let's use `mygrad` to compute the derivative of
![CodeCogsEqn.gif](https://user-images.githubusercontent.com/29104956/39901776-9e5ed362-5498-11e8-9890-e84aa2b6dae1.gif),
which is <a href="https://www.codecogs.com/eqnedit.php?latex=df/dx&space;=&space;2x" target="_blank"><img
src="https://latex.codecogs.com/gif.latex?df/dx&space;=&space;2x" title="\frac{df}{dx} = 2x"
/></a>.

`mygrad.Tensor` behaves nearly identically to NumPy's ndarray, in addition to having the machinery needed to
compute the analytic derivatives of functions. Suppose we want to compute this derivative at `x = 3`. We can create a
0-dimensional tensor (a scalar) for x and compute `f(x)`:

```python
>>> import mygrad as mg
>>> x = mg.Tensor(3.0)
>>> f = x ** 2
>>> f
Tensor(9.0)
```

Invoking `f.backward()` instructs `mygrad` to trace through the computational graph that produced `f` and compute the
derivatives of `f` with respect to all of its independent variables. Thus, executing `f.backward()` will compute <a
href="https://www.codecogs.com/eqnedit.php?latex=df/dx&" target="_blank"><img
src="https://latex.codecogs.com/gif.latex?df/dx&" title="\frac{df}{dx}" /></a> and will store the value in
`x.grad`:

```python
>>> f.backward()  # triggers computation of `df/dx`
>>> x.grad  # df/dx = 2x = 6.0
array(6.0)
```

This is the absolute tip of the iceberg. `mygrad` can compute derivatives of multivariable composite
functions of tensor-valued variables!

## Some Bells and Whistles
`mygrad` supports all of NumPy's essential features, including:
 - [N-dimensional tensors](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/IntroducingTheNDarray.html) that can be reshaped and have their axes transposed
 - [vectorization](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/VectorizedOperations.html)
 - [broadcasting](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html)
 - [basic and advanced indexing](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html) (including all varieties of mixed indexing schemes) for both getting and setting items.
 - fully-fledged support for [einsum](https://rockt.github.io/2018/04/30/einsum) (including broadcasting and traces,
   which are not supported by PyTorch, TensorFlow, or HIPS-autograd)

 `mygrad.Tensor` plays nicely with NumPy-arrays, which behave as constants when they are used in computational graphs:

```python
>>> import numpy as np
>>> x = mg.Tensor([2.0, 2.0, 2.0])
>>> y = np.array([1.0, 2.0, 3.0])
>>> f = x ** y  # (2 ** 1, 2 ** 2, 2 ** 3)
>>> f.backward()
>>> x.grad
array([ 1.,  4., 12.])
```

`mygrad.nnet` supplies essential functions for machine learning, including:
- N-dimensional convolutions (with striding, dilation, and padding)
- N-dimensional pooling
- A [gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) for sequence-learning (with input-level
  dropout and variational hidden-hidden dropout)

It leverages a nice [sliding window
view](https://github.com/rsokl/MyGrad/blob/a72ebc26acf5c254f59a562c8045698387763a41/mygrad/nnet/layers/utils.py#L6)
function, which produces convolution-style windowed views of arrays/tensors without making copies of them, to
intuitively (and quite efficiently) perform the neural network-style convolutions and pooling.

## Advanced Example
The following is an example of using `mygrad` to compute the [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) of classification scores and to "backpropagate" through (compute the gradient of) this loss. This example demonstrates some of mygrad's ability to perform backpropagation through broadcasted operations, basic indexing, advanced indexing, and in-place assignments.

```python
>>> from mygrad import Tensor
>>> import numpy as np
>>> class_scores = Tensor(10 * np.random.rand(100, 10))         # 100 samples, 10 possible classes for each
>>> class_labels = np.random.randint(low=0, high=10, size=100)  # correct label for each datum
>>> class_labels = (range(len(class_labels)), class_labels)
>>> correct_class_scores = class_scores[class_labels]

>>> Lij = class_scores - correct_class_scores[:, np.newaxis] + 1.  # 100x10 margins
>>> Lij[Lij <= 0] = 0      # scores within the hinge incur no loss
>>> Lij[class_labels] = 0  # the score corresponding to the correct label incurs no loss

>>> loss = Lij.sum() / class_scores.shape[0]  # compute mean hinge loss
>>> loss.backward()    # compute gradient of loss w.r.t all dependent tensors
>>> class_scores.grad  # d(loss)/d(class_scores)
array([[ 0.  ,  0.01,  0.  , -0.04,  0.  ,  0.  ,  0.01,  0.  ,  0.01, 0.01], ...])
```

## Computational Graph Visualization
`mygrad` uses [Graphviz](http://www.graphviz.org) and a [Python interface for Graphviz](https://graphviz.readthedocs.io/en/stable/) to render the computational graphs built using tensors. These graphs can be rendered in Jupyter notebooks, allowing for quick checks of graph structure, or can be saved to file for later reference.

The dependencies can be installed with:

```shell
conda install graphviz
conda install python-graphviz
```
