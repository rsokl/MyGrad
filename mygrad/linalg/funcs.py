from .ops import *
from mygrad.tensor_base import Tensor
from numpy.core.einsumfunc import _parse_einsum_input

__all__ = ["matmul", "einsum"]


def matmul(a, b, constant=False):
    """
    Matrix product of two tensors:

    ``matmul(x, y)`` is equivalent to ``x @ y``.

    This documentation was adapted from ``numpy.matmul``

    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional
      matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of
      matrices residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by
      prepending a 1 to its dimensions. After matrix multiplication
      the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by
      appending a 1 to its dimensions. After matrix multiplication
      the appended 1 is removed.

    Multiplication by a scalar is not allowed, use ``*`` instead. Note that
    multiplying a stack of matrices with a vector will result in a stack of
    vectors, but matmul will not recognize it as such.

    ``matmul`` differs from ``numpy.dot`` in two important ways.

    - Multiplication by scalars is not allowed.
    - Stacks of matrices are broadcast together as if the matrices
      were elements.


    Parameters
    ----------
    a : array_like
    
    b : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    output : mygrad.Tensor
        Returns the matrix product of `a` and `b`.  If `a` and `b` are both
        1-D arrays then a scalar is returned; otherwise an array is
        returned.


    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

        If scalar value is passed.

    Notes
    -----
    The matmul function implements the semantics of the `@` operator introduced
    in Python 3.5 following PEP465.

    Examples
    --------
    For two 2D tensors, ``matmul(a, b)`` is the matrix product :math:`\sum_{j}{A_{ij} B_{jk}} = F_{ik}`:

    >>> import mygrad as mg
    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> mg.matmul(a, b)
    Tensor([[4, 1],
            [2, 2]])

    For 2-D mixed with 1-D, the result is the matrix-vector product, :math:`\sum_{j}{A_{ij} B_{j}} = F_{i}`:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [1, 2]
    >>> mg.matmul(a, b)
    Tensor([1, 2])

    Broadcasting is conventional for stacks of arrays. Here ``a`` is treated
    like a stack of three 5x6 matrices, and the 6x4 matrix ``b`` is broadcast
    matrix-multiplied against each one. This produces a shape-(3, 5, 4) tensor
    as a result.

    >>> a = mg.arange(3*5*6).reshape((3,5,6))
    >>> b = mg.arange(6*4).reshape((6,4))
    >>> mg.matmul(a,b).shape
    (3, 5, 4)
    >>> mg.matmul(a,b)[0,1,1]
    Tensor(98)
    >>> mg.sum(a[0,1,:] * b[0,:,1])
    Tensor(98)

    Scalar multiplication raises an error.

    >>> mg.matmul([1,2], 3)
    Traceback (most recent call last):
    ...
    ValueError: Scalar operands are not allowed, use '*' instead"""
    return Tensor._op(MatMul, a, b, constant=constant)


def einsum(*operands, optimize=False, constant=False):
    """
    einsum(subscripts, *operands)

    Evaluates the Einstein summation convention on the operands. This implementation
    exactly mirrors that of ``numpy.einsum`` and supports back-propagation through
    all variety of tensor-products, sums, traces, and views that it can perform.

    The following docstring was adapted from the documentation for ``numpy.einsum``

    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion.  This function
    provides a way to compute such summations. The best way to understand this
    function is to try the examples below, which show how many common NumPy/MyGrad
    functions can be implemented as calls to ``einsum``.

    Back-propagation via ``einsum`` is optimized such that any tensor that occurs
    redundantly within the summation will only have its gradient computed once.
    This optimization accommodates all number and combination of redundancies that can
    be encountered.

    E.g. back-propping through ``einsum('...,...->', x, x)`` will only incur a single
    computation/accumulation for ``x.grad`` rather than two. This permits users to
    leverage the efficiency of sum-reduction, where ``(x ** 2).sum()`` is sub-optimal,
    without being penalized during back-propagation.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.

    operands : array_like
        The tensors used in the summation.

    optimize : {False, True, 'greedy', 'optimal'}, optional (default=False)
        Controls if intermediate optimization should occur; also enables
        the use of BLAS where possible. This can produce significant speedups
        for computations like matrix multiplication.

        No optimization will occur if False and True will default to the 'greedy'
        algorithm. Also accepts an explicit contraction list from the
        ``np.einsum_path`` function. See ``np.einsum_path`` for more details.

    constant : bool, optional (default=False)
        If True, the resulting Tensor is a constant.

    Returns
    -------
    output : mygrad.Tensor
        The calculation based on the Einstein summation convention.

    Notes
    -----
    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Repeated subscripts labels in one operand take the diagonal.  For example,
    ``einsum('ii', a)`` is equivalent to ``np.trace(a)`` (however, the former
    supports back-propagation).

    Whenever a label is repeated, it is summed, so ``einsum('i, i', a, b)``
    is equivalent to ``np.inner(a, b)``.  If a label appears only once,
    it is not summed, so ``einsum('i', a)`` produces a view of ``a``
    with no changes.

    The order of labels in the output is by default alphabetical.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D tensor, while
    ``einsum('ji', a)`` takes its transpose.

    The output can be controlled by specifying output subscript labels
    as well.  This specifies the label order, and allows summing to
    be disallowed or forced when desired.  The call ``einsum('i->', a)``
    is like ``np.sum(a, axis=-1)``, and ``einsum('ii->i', a)``
    is like ``np.diag(a)``.  The difference is that `einsum` does not
    allow broadcasting by default.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``einsum('...ii->...i', a)``.
    To take the trace along the first and last axes,
    you can do ``einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, you can do
    ``einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new tensor.  Thus, taking the diagonal as ``einsum('ii->i', a)``
    produces a view.

    An alternative way to provide the subscripts and operands is as
    ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``. The examples
    below have corresponding `einsum` calls with the two parameter methods.

    Examples
    --------
    >>> import mygrad as mg
    >>> import numpy as np
    >>> a = mg.arange(25).reshape(5,5)
    >>> b = mg.arange(5)
    >>> c = mg.arange(6).reshape(2,3)

    Compute the trace of ``a``, :math:`\sum_{i}{A_{ii}} = f`:

    >>> einsum('ii', a)
    Tensor(60)
    >>> einsum(a, [0, 0])
    Tensor(60)
    >>> np.trace(a.data)
    array(60)

    Return a view along the diagonal of ``a``, :math:`A_{ii} = F_{i}`:

    >>> einsum('ii->i', a)
    Tensor([ 0,  6, 12, 18, 24])
    >>> einsum(a, [0,0], [0])
    Tensor([ 0,  6, 12, 18, 24])
    >>> np.diag(a.data)
    array([ 0,  6, 12, 18, 24])

    Compute the matrix-vector product of ``a`` with ``b``, :math:`\sum_{j}{A_{ij} B_{j}} = F_{i}`:

    >>> einsum('ij,j', a, b)
    Tensor([ 30,  80, 130, 180, 230])
    >>> einsum(a, [0,1], b, [1])
    Tensor([ 30,  80, 130, 180, 230])
    >>> mg.matmul(a, b)
    Tensor([ 30,  80, 130, 180, 230])
    >>> einsum('...j,j', a, b)
    Tensor([ 30,  80, 130, 180, 230])

    Take the transpose of ``c``, :math:`C_{ji} = F_{ij}`:

    >>> einsum('ji', c)
    Tensor([[0, 3],
            [1, 4],
            [2, 5]])
    >>> einsum(c, [1, 0])
    Tensor([[0, 3],
            [1, 4],
            [2, 5]])
    >>> c.T
    Tensor([[0, 3],
            [1, 4],
            [2, 5]])

    Compute ``3 * c``:

    >>> einsum('..., ...', 3, c)
    Tensor([[ 0,  3,  6],
            [ 9, 12, 15]])
    >>> einsum(',ij', 3, c)
    Tensor([[ 0,  3,  6],
            [ 9, 12, 15]])
    >>> einsum(3, [Ellipsis], c, [Ellipsis])
    Tensor([[ 0,  3,  6],
            [ 9, 12, 15]])
    >>> 3 * c
    Tensor([[ 0,  3,  6],
            [ 9, 12, 15]])

    Compute the inner product of ``b`` with itself, :math:`\sum_{i}{B_{i} B_{i}} = f`:

    >>> einsum('i,i', b, b)
    Tensor(30)
    >>> einsum(b, [0], b, [0])
    Tensor(30)
    >>> np.inner(b.data, b.data)
    30

    Compute the outer product of ``array([1, 2])`` with ``b``, :math:`A_{i}B_{j} = F_{ij}`:

    >>> einsum('i,j', np.arange(2)+1, b)
    Tensor([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> einsum(np.arange(2)+1, [0], b, [1])
    Tensor([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> einsum('i...->...', a)
    Tensor([50, 55, 60, 65, 70])
    >>> einsum(a, [0,Ellipsis], [Ellipsis])
    Tensor([50, 55, 60, 65, 70])
    >>> np.sum(a, axis=0)
    array([50, 55, 60, 65, 70])

    Compute the tensor product :math:`\sum_{ij}{A_{ijk} B_{jil}} = F_{kl}`

    >>> a = mg.arange(60.).reshape(3,4,5)
    >>> b = mg.arange(24.).reshape(4,3,2)
    >>> einsum('ijk,jil->kl', a, b)
    Tensor([[ 4400.,  4730.],
            [ 4532.,  4874.],
            [ 4664.,  5018.],
            [ 4796.,  5162.],
            [ 4928.,  5306.]])
    >>> einsum(a, [0,1,2], b, [1,0,3], [2,3])
    Tensor([[ 4400.,  4730.],
            [ 4532.,  4874.],
            [ 4664.,  5018.],
            [ 4796.,  5162.],
            [ 4928.,  5306.]])
    >>> np.tensordot(a,b, axes=([1,0],[0,1]))
    array([[ 4400.,  4730.],
            [ 4532.,  4874.],
            [ 4664.,  5018.],
            [ 4796.,  5162.],
            [ 4928.,  5306.]])

    Matrix multiply ``a.T`` with ``b.T``, :math:`\sum_{k}{A_{ki} B_{jk}} = F_{ij}`

    >>> a = mg.arange(6).reshape((3,2))
    >>> b = mg.arange(12).reshape((4,3))
    >>> einsum('ki,jk->ij', a, b)
    Tensor([[10, 28, 46, 64],
            [13, 40, 67, 94]])
    >>> einsum('ki,...k->i...', a, b)
    Tensor([[10, 28, 46, 64],
            [13, 40, 67, 94]])
    >>> einsum('k...,jk', a, b)
    Tensor([[10, 28, 46, 64],
            [13, 40, 67, 94]])

    Make an assignment to a view along the diagonal of ``a``:

    >>> a = mg.zeros((3, 3))
    >>> einsum('ii->i', a).data[:] = 1
    >>> a
    Tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    """

    # TODO: normalize error handling for invalid inputs
    operands = list(operands)
    if isinstance(operands[0], str):
        # operands form: "ijk, ijk", x, y
        variables = operands[1:]
        if any(isinstance(i, Tensor) for i in operands):
            operands[1:] = (var.data if isinstance(var, Tensor) else var for var in operands[1:])
    else:
        # operands form: op0, sublist0, op1, sublist1, ..., [sublistout]
        end = -1 if len(operands) % 2 else None  # -1 if sublistout is included
        variables = operands[:end:2]
        if any(isinstance(i, Tensor) for i in operands):
            operands[:end:2] = (var.data if isinstance(var, Tensor) else var for var in operands[:end:2])

    in_lbls, out_lbls, _ = _parse_einsum_input(operands)
    return Tensor._op(EinSum, *variables, op_kwargs=dict(in_lbls=in_lbls,
                                                         out_lbls=out_lbls,
                                                         optimize=optimize),
                      constant=constant)
