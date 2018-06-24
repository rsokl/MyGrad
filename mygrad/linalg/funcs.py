from .ops import *
from mygrad.tensor_base import Tensor

__all__ = ["matmul"]

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

    ``matmul`` differs from ``dot`` in two important ways.

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