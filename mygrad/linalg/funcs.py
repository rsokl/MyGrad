from .ops import *
from mygrad.tensor_base import Tensor
from numpy.core.einsumfunc import _parse_einsum_input
import numpy as np

__all__ = ["multi_matmul", "matmul", "einsum"]


def multi_matmul(arrays, constant = False):
    """
    Matrix product of two or more arrays calculated in the optimal ordering

    Parameters
    ----------
    arrays: sequence or array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        Returns the matrix product of the arrays provided


	Extended Summary
    ----------------
    This documentation was adapted from ``numpy.linalg.multi_dot``

    Compute the matrix multiplication of two or more arrays in a single function 
    call, while automatically selecting the fastest evaluation order.
    `multi_matmul` chains `matmul` and uses optimal parenthesization (see notes).
    Depending on the shapes of the matrices, this can speed up the multiplication a lot.
    If the first argument is 1-D it is treated as a row vector.
    If the last argument is 1-D it is treated as a column vector.
    The other arguments must be 2-D or greater.

    Raises
    ------
    ValueError
        If arrays contains less than two array_like items.

    ValueError
    	If array other than the first or last is less than two dimmensional

    
    Notes
    -----
    The cost for a matrix multiplication can be calculated with the
    following function::
        def cost(A, B):
            return A.shape[0] * A.shape[1] * B.shape[1]
    Let's assume we have three matrices
    :math:`A_{10x100}, B_{100x5}, C_{5x50}`.
    The costs for the two different parenthesizations are as follows::
        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000
    """

    n = len(arrays)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return matmul(arrays[0], arrays[1], constant)

    arrays = [np.asanyarray(a) for a in arrays]
    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of this function simpler
    if arrays[0].ndim == 1:
        arrays[0] = arrays[0][np.newaxis,:]
    if arrays[-1].ndim == 1:
        arrays[-1] = arrays[-1][:,np.newaxis]

    for a in arrays:
        if a.ndim < 2:
            raise ValueError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)

    if n == 3:
        result = _multi_matmul_three(arrays[0], arrays[1], arrays[2], constant)
    else:
        order = _multi_matmul_chain_order(arrays)
        result = _multi_matmul(arrays, order, 0, n - 1, constant)

    # return proper shape since we possibly added dimmensions to the first
    # and last arrays
    #if ndim_first == 1 and ndim_last == 1:
    #    return result[0, 0]
    #elif ndim_first == 1 or ndim_last == 1:
    #    return result.ravel()
    #else:
        return result


def _multi_matmul_three(A, B, C, constant = False):
    """
    Find the best order for three arrays and do the multiplication.
    
    """
    a0, a1b0 = A.shape[-2:]
    b1c0, c1 = C.shape[-2:]
    cost1 = a0 * b1c0 * (a1b0 + c1)
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return matmul(matmul(A, B, constant), C, constant)
    else:
        return matmul(A, matmul(B, C, constant), constant)


def _multi_matmul_chain_order(arrays):
    """
    Return a np.array that encodes the optimal order of mutiplications.
    The optimal order array is then used by `_multi_matmul()` to do the
    multiplication.
    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])
    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    # Using -2 to generalize for shapes that are more than 2 dimmensions
    p = [a.shape[-2] for a in arrays] + [arrays[-1].shape[-1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = np.zeros((n, n), dtype=np.double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = np.empty((n, n), dtype=np.intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = np.inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return s


def _multi_matmul(arrays, order, i, j, constant = False):
    """Actually do the multiplication with the given order."""
    if i == j:
        return arrays[i]
    else:
        return matmul(_multi_matmul(arrays, order, i, order[i, j], constant),
                _multi_matmul(arrays, order, order[i, j] + 1, j, constant), constant)


def matmul(a, b, constant=False):
    """ 
    ``f(a, b) -> matmul(a, b)``

    Matrix product of two arrays.

    Parameters
    ----------
    a : array_like
    
    b : array_like

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        Returns the matrix product of `a` and `b`.  If `a` and `b` are both
        1-D arrays then a scalar is returned; otherwise an array is
        returned.

    
    Extended Summary
    ----------------
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
    For 2-D tensores it is the matrix product:

    >>> import mygrad as mg
    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> mg.matmul(a, b)
    Tensor([[4, 1],
            [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = [[1, 0], [0, 1]]
    >>> b = [1, 2]
    >>> mg.matmul(a, b)
    Tensor([1, 2])
    >>> mg.matmul(b, a)
    Tensor([1, 2])


    Broadcasting is conventional for stacks of arrays

    >>> a = mg.arange(2*2*4).reshape((2,2,4))
    >>> b = mg.arange(2*2*4).reshape((2,4,2))
    >>> mg.matmul(a,b).shape
    (2, 2, 2)
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

    The following docstring was adapted from the documentation for ``numpy.einsum``

    Evaluates the Einstein summation convention on the operands.
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
    ``einsum('ii', a)`` is equivalent to ``np.trace(a)``.

    Whenever a label is repeated, it is summed, so ``einsum('i,i', a, b)``
    is equivalent to ``np.inner(a,b)``.  If a label appears only once,
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
    
    >>> einsum('ii', a)  # the trace of a
    Tensor(0)
    >>> einsum(a, [0, 0])
    Tensor(60)
    >>> np.trace(a.data)
    array(60)

    >>> einsum('ii->i', a)  # view along diagonal of a
    Tensor([ 0,  6, 12, 18, 24])
    >>> einsum(a, [0,0], [0])
    Tensor([ 0,  6, 12, 18, 24])
    >>> np.diag(a.data)
    array([ 0,  6, 12, 18, 24])

    >>> einsum('ij,j', a, b)
    Tensor([ 30,  80, 130, 180, 230])
    >>> einsum(a, [0,1], b, [1])
    Tensor([ 30,  80, 130, 180, 230])
    >>> mg.matmul(a, b)
    Tensor([ 30,  80, 130, 180, 230])
    >>> einsum('...j,j', a, b)
    Tensor([ 30,  80, 130, 180, 230])

    >>> einsum('ji', c)  # transpose of c
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

    >>> einsum('i,i', b, b)
    Tensor(30)
    >>> einsum(b, [0], b, [0])
    Tensor(30)
    >>> np.inner(b.data, b.data)
    30

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
