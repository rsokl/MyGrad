# Introducing mygrad
`mygrad` is a simple, NumPy-centric autograd library (automatically computes derivatives of mathematical functions). It
is designed to serve primarily as an education tool; it is easy to install, has a readable and easily customizable code
base, and provides a sleek interface that mimics NumPy. Furthermore, it leverages NumPy's vectorization
to achieve good performance despite the library's simplicity. 

This is not meant to be a competitor to libraries like PyTorch (which `mygrad` most closely resembles) or
TensorFlow. Rather, it is meant to serve as a useful tool for students who are learning about training neural networks
using back propagation.

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
>>> x.grad  # df/dx = 2x = 6.
array(6.)
```

This is the absolute tip of the iceberg. `mygrad` can compute derivatives of multivariable, composite/recursive
functions of tensor-valued variables!

## Some Bells and Whistles
`mygrad` supports all of NumPy's essential features, including:
 - [N-dimensional tensors](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/IntroducingTheNDarray.html) that can be reshaped and have their axes transposed
 - [vectorization](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/VectorizedOperations.html)
 - [broadcasting](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html)
 - [basic and advanced indexing](http://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/BasicIndexing.html) (including all varieties of mixed indexing schemes) for "getting" and "setting" items from/to tensors
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

`mygrad.nn` supplies essential functions for machine learning, including:
- N-dimensional convolutions (with striding, dilation, and padding)
- N-dimensional pooling
- A [gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) for sequence-learning (with input-level
  dropout and variational hidden-hidden dropout)

It leverages a nice [sliding window
view](https://github.com/rsokl/MyGrad/blob/a72ebc26acf5c254f59a562c8045698387763a41/mygrad/nnet/layers/utils.py#L6)
function, which produces convolution-style windowed views of arrays/tensors without making copies of them, to
intuitively (and quite efficiently) perform the neural network-style convolutions and pooling.

