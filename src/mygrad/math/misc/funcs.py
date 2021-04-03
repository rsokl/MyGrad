from typing import Optional, Union

import numpy as np
from numpy import ndarray

import mygrad as mg
from mygrad.math.misc.ops import MatMul
from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike, DTypeLikeReals, Mask
from mygrad.ufuncs import ufunc_creator

from .ops import Abs, Cbrt, Maximum, Minimum, Sqrt

__all__ = [
    "abs",
    "absolute",
    "cbrt",
    "clip",
    "sqrt",
    "maximum",
    "minimum",
    "matmul",
    "multi_matmul",
]


@ufunc_creator(Abs)
def absolute(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """The absolute value, computed elementwise.

    This docstring was adapted from that of numpy.absolute [1]_

    Parameters
    ----------
    x : ArrayLike
        Input array.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    absolute : Tensor
        An ndarray containing the absolute value of
        each element in `x`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.absolute.html

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.array([-1.2, 1.2])
    >>> mg.absolute([-1.2, 1.2])
    Tensor([ 1.2,  1.2])

    Plot the function and its derivate over ``[-10, 10]``:

    .. plot::

       >>> import mygrad as mg
       >>> import matplotlib.pyplot as plt
       >>> x = mg.linspace(-5, 5, 100)
       >>> y = mg.absolute(x)
       >>> plt.title("absolute(x)")
       >>> y.backward()
       >>> plt.plot(x, x.grad, label="df/dx")
       >>> plt.plot(x, y, label="f(x)")
       >>> plt.legend()
       >>> plt.grid()
       >>> plt.show()
    """
    ...


abs = absolute


@ufunc_creator(Sqrt)
def sqrt(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """The square root, elementwise.

    This docstring was adapted from that of numpy.sqrt [1]_

    Parameters
    ----------
    x : ArrayLike
        The values whose square-roots are required.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.


    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    y : ndarray
        A tensor of the same shape as `x`, containing the positive
        square-root of each element in `x`. Negative-valued inputs
        produce nans.


    Notes
    -----
    *sqrt* has--consistent with common convention--as its branch cut the
    real "interval" [`-inf`, 0), and is continuous from above on it.
    A branch cut is a curve in the complex plane across which a given
    complex function fails to be continuous.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.sqrt([1, 4, 9])
    Tensor([ 1.,  2.,  3.])

    >>> mg.sqrt([4, -1, mg.inf])
    Tensor([ 2., nan, inf])
    """
    ...


@ufunc_creator(Cbrt)
def cbrt(
    x: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """The cube root elementwise.

    This docstring was adapted from that of numpy.cbrt [1]_

    Parameters
    ----------
    x : ArrayLike
        The values whose cube-roots are computed.

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    y : ndarray
        A tensor of the same shape as `x`, containing the cube
        cube-root of each element in `x`.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.cbrt([1, 8, 27])
    Tensor([ 1.,  2.,  3.])
    """
    ...


@ufunc_creator(Maximum)
def maximum(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Pair-wise maximum of tensor elements.

    This docstring was adapted from that of numpy.maximum [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        The tensors holding the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    y : Tensor
        The maximum of `x1` and `x2`, element-wise.

    See Also
    --------
    minimum :
        Element-wise minimum of two arrays, propagates NaNs.

    Notes
    -----
    The maximum is equivalent to ``mg.where(x1 >= x2, x1, x2)`` when
    neither x1 nor x2 are nans, but it is faster and does proper
    broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.maximum.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.maximum([2, 3, 4], [1, 5, 2])
    Tensor([2, 5, 4])

    >>> mg.maximum(mg.eye(2), [0.5, 2]) # broadcasting
    Tensor([[ 1. ,  2. ],
           [ 0.5,  2. ]])

    >>> mg.maximum([mg.nan, 0, mg.nan], [0, mg.nan, mg.nan])
    Tensor([nan, nan, nan])
    >>> mg.maximum(mg.Inf, 1)
    Tensor(inf)
    """
    ...


@ufunc_creator(Minimum)
def minimum(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[ndarray, Tensor]] = None,
    *,
    where: Mask = True,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    """Pair-wise minimum of tensor elements.

    This docstring was adapted from that of numpy.minimum [1]_

    Parameters
    ----------
    x1, x2 : ArrayLike
        The tensors holding the elements to be compared.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).

    out : Optional[Union[Tensor, ndarray]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    where : Mask
        This condition is broadcast over the input. At locations where the
        condition is True, the ``out`` tensor will be set to the ufunc result.
        Elsewhere, the ``out`` tensor will retain its original value.
        Note that if an uninitialized `out` tensor is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    Returns
    -------
    y : Tensor
        The minimum of `x1` and `x2`, element-wise.

    See Also
    --------
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.

    Notes
    -----
    The minimum is equivalent to ``mg.where(x1 <= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster and does proper
    broadcasting.

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.minimum.html

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.minimum([2, 3, 4], [1, 5, 2])
    Tensor([1, 3, 2])

    >>> mg.minimum(mg.eye(2), [0.5, 2]) # broadcasting
    Tensor([[ 0.5,  0. ],
           [ 0. ,  1. ]])

    >>> mg.minimum([mg.nan, 0, mg.nan],[0, mg.nan, mg.nan])
    Tensor([nan, nan, nan])
    >>> mg.minimum(-mg.Inf, 1)
    Tensor(-inf)
    """
    ...


@implements_numpy_override()
def clip(
    a: ArrayLike, a_min: ArrayLike, a_max: ArrayLike, *, constant: Optional[bool] = None
) -> Tensor:
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Equivalent to `mg.minimum(a_max, mg.maximum(a, a_min))``.

    No check is performed to ensure ``a_min < a_max``.

    This docstring was adapted from that of `numpy.clip`

    Parameters
    ----------
    a : ArrayLike
        Array containing elements to clip.

    a_min : Optional[float, ArrayLike]
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.

    a_max : Optional[float, ArrayLike]
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`. If `a_min` or `a_max` are ArrayLike, then the three
        arrays will be broadcasted to match their shapes.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not backpropagate a gradient)

    Returns
    -------
    Tensor
        A tensor with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Examples
    --------
    >>> import mygrad as mg
    >>> a = mg.arange(10)
    >>> mg.clip(a, 1, 8)
    Tensor([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a
    Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mg.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    Tensor([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])"""
    if a_min is None and a_max is None:
        raise ValueError("`a_min` and `a_max` cannot both be set to `None`")

    if a_min is not None:
        a = maximum(a_min, a, constant=constant)

    if a_max is not None:
        a = minimum(a_max, a, constant=constant)

    return a


@ufunc_creator(MatMul)
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    out: Optional[Union[np.ndarray, Tensor]] = None,
    *,
    dtype: DTypeLikeReals = None,
    constant: Optional[bool] = None,
) -> Tensor:  # pragma: no cover
    r"""
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
    x1 : ArrayLike

    x2 : ArrayLike

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    dtype : Optional[DTypeLikeReals]
        The dtype of the resulting tensor.

    out : Optional[Union[ndarray, Tensor]]
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated tensor is returned.

    Returns
    -------
    output : mygrad.Tensor
        Returns the matrix product of ``x1`` and `x2``.

    Raises
    ------
    ValueError
        If :
         - The last dimension of ``x1`` is not the same size as
           the second-to-last dimension of ``x2``.
         - If scalar value is passed.

    See Also
    --------
    einsum : Einstein summation convention.

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

    Scalar multiplication raises an error.

    >>> mg.matmul(a, 3)
    Traceback (most recent call last):
    ...
    ValueError: Scalar operands are not allowed, use '*' instead"""
    ...


def multi_matmul(tensors: ArrayLike, *, constant: Optional[bool] = None) -> Tensor:
    """
    Matrix product of two or more tensors calculated in the optimal ordering

    Parameters
    ----------
    tensors: Sequence[array_like]
        The sequence of tensors to be matrix-multiplied.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    mygrad.Tensor
        Returns the matrix product of the tensors provided


    Extended Summary
    ----------------
    This documentation was adapted from ``numpy.linalg.multi_dot``

    Compute the matrix multiplication of two or more arrays in a single function
    call, while automatically selecting the fastest evaluation order.
    ``multi_matmul`` chains ``matmul`` and uses optimal parenthesization  [1]_ [2]_.
    Depending on the shapes of the matrices, this can speed up the multiplication a lot.

    If the first argument is 1-D it is treated as a row vector.

    If the last argument is 1-D it is treated as a column vector.

    The other arguments must be 2-D or greater.

    Think of `multi_dot` as an optimized version of::

        def multi_dot(tensors): return functools.reduce(mg.matmul, tensors)

    Raises
    ------
    ValueError
        If ``tensors`` contains less than two array_like items.

    ValueError
        If ``tensor`` other than the first or last is less than two dimensional

    See Also
    --------
    matmul : matrix multiplication with two arguments.

    References
    ----------

    .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
    .. [2] http://en.wikipedia.org/wiki/Matrix_chain_multiplication

    Notes
    -----
    The cost for a matrix multiplication can be calculated with the
    following function::

        def cost(A, B):
            return A.shape[0] * A.shape[1] * B.shape[1]

    Let's assume we have three matrices :math:`A_{10x100}, B_{100x5}, C_{5x50}`.

    The costs for the two different parenthesizations are as follows::

        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

    Examples
    --------
    ``multi_matmul`` allows you to write:

    >>> from mygrad.math.misc.funcs import matmul    >>> from mygrad import multi_matmul,  Tensor
    >>> import numpy as np
    >>> # Prepare some random tensors
    >>> A = Tensor(np.random.random((10000, 100)))
    >>> B = Tensor(np.random.random((100, 1000)))
    >>> C = Tensor(np.random.random((1000, 5)))
    >>> D = Tensor(np.random.random((5, 333)))
    >>> # the actual matrix multiplication
    >>> multi_matmul([A, B, C, D]) # computes (A @ (B @ C)) @ D

    instead of:

    >>> matmul(matmul(matmul(A, B), C), D)
    >>> # or
    >>> A @ B @ C @ D
    """

    for a in tensors:
        if not (1 <= a.ndim <= 2):
            raise ValueError(
                "%d-dimensional tensor given. Tensor must be one or two-dimensional"
                % (a.ndim,)
            )

    n = len(tensors)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return matmul(tensors[0], tensors[1], constant=constant)

    tensors = [a if isinstance(a, Tensor) else np.asarray(a) for a in tensors]

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = tensors[0].ndim, tensors[-1].ndim

    # Explicitly convert vectors to 2D arrays to keep the logic of this function simpler
    if tensors[0].ndim == 1:
        tensors[0] = mg.expand_dims(
            tensors[0],
            axis=0,
            constant=tensors[0].constant if isinstance(tensors[0], Tensor) else True,
        )
    if tensors[-1].ndim == 1:
        tensors[-1] = mg.expand_dims(
            tensors[-1],
            axis=1,
            constant=tensors[-1].constant if isinstance(tensors[-1], Tensor) else True,
        )

    if n == 3:
        result = _multi_matmul_three(
            tensors[0], tensors[1], tensors[2], constant=constant
        )
    else:
        order = _multi_matmul_chain_order(tensors)
        result = _multi_matmul(tensors, order, 0, n - 1, constant=constant)

    # return proper shape since we possibly added dimensions to the first
    # and last arrays
    if ndim_first == 1 and ndim_last == 1:
        result = result[0, 0]
        return result
    elif ndim_first == 1 or ndim_last == 1:
        result = result.reshape(-1)
        return result
    else:
        return result


def _multi_matmul_three(A, B, C, *, constant=None) -> Tensor:
    """
    Find the best order for three arrays and do the multiplication.

    """
    a0, a1b0 = A.shape[-2:]
    b1c0, c1 = C.shape[-2:]
    cost1 = a0 * b1c0 * (a1b0 + c1)
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return matmul(matmul(A, B, constant=constant), C, constant=constant)
    else:
        return matmul(A, matmul(B, C, constant=constant), constant=constant)


def _multi_matmul_chain_order(arrays):
    """
    Return a np.array that encodes the optimal order of multiplications.
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

    for ind in range(1, n):
        for i in range(n - ind):
            j = i + ind
            m[i, j] = np.inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index
    return s


def _multi_matmul(arrays, order, i, j, *, constant=None) -> Tensor:
    """Actually do the multiplication with the given order."""
    if i == j:
        return arrays[i]
    else:
        return matmul(
            _multi_matmul(arrays, order, i, order[i, j], constant=constant),
            _multi_matmul(arrays, order, order[i, j] + 1, j, constant=constant),
            constant=constant,
        )
