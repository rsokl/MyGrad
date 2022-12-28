from numbers import Real
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.core.einsumfunc import _parse_einsum_input

from mygrad.math.misc.funcs import absolute
from mygrad.math.sequential.funcs import max as mg_max, min as mg_min
from mygrad.tensor_base import Tensor, implements_numpy_override
from mygrad.typing import ArrayLike

from .ops import EinSum, Norm

__all__ = ["einsum", "norm"]


@implements_numpy_override(np.linalg.norm)
def norm(
    x: ArrayLike,
    ord: Optional[Union[int, float]] = None,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    *,
    nan_to_num: bool = True,
    constant: Optional[bool] = None,
) -> Tensor:
    r"""Vector norm.

    This function is an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    In contrast to ``numpy.linalg.norm``, matrix norms are not supported.

    This docstring was adapted from that of ``numpy.linalg.norm`` [1]_.

    Parameters
    ----------
    x : ArrayLike
        Input tensor.  If `axis` is None, then `x` must be 1-D unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.

    ord : Optional[Union[int, float]]
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.

    axis : Optional[Union[int, Tuple[int]]]
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms. The default is None.

    keepdims : bool, optional (default=False)
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    nan_to_num : bool, optional (default=True)
        If `True` then gradients that would store nans due to the presence of
        zeros in `x` will instead store zeros in those places.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

    Returns
    -------
    Tensor
        Norm(s) of the vector(s).

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ==========================
    ord    norm for vectors
    =====  ==========================
    inf    max(abs(x))
    -inf   min(abs(x))
    0      sum(x != 0)
    1      as below
    -1     as below
    2      as below
    -2     as below
    other  sum(abs(x)**ord)**(1./ord)
    =====  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] Retrived from: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    .. [2] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.tensor([[1.0, 2.0, 3.0],
    ...                [1.0, 0.0, 0.0]])
    >>> l2_norms = mg.linalg.norm(x, axis=1, ord=2)
    >>> l2_norms
    Tensor([3.74165739, 1.        ])

    The presence of the elementwise absolute values in the norm operation means that zero-valued entries in any of
    input vectors have an undefined derivative. When `nan_to_num=False` is specified these derivatives will be reported
    as `nan`, otherwise they will be made to be 0.0.

    >>> l2_norms = mg.linalg.norm(x, axis=1, ord=2, nan_to_num=True)
    >>> l2_norms.backward()
    >>> x.grad
    array([[0.26726124, 0.53452248, 0.80178373],
           [1.        ,        nan,        nan]])

    This is rigorously true, but is often not the desired behavior in autodiff applications.
    Rather, it can be preferable to use `0.0` to fill these undefined derivatives.
    This is the default behavior, when `nan_to_num` is not specified.

    >>> l2_norms = mg.linalg.norm(x, axis=1, ord=2, nan_to_num=False)  # default setting: `nan_to_num=False`
    >>> l2_norms.backward()
    >>> x.grad
    array([[0.26726124, 0.53452248, 0.80178373],
          [1.        ,          0.,         0.]])

    L1 norms along each of the three columns:

    >>> mg.linalg.norm(x, axis=0, ord=1)
    Tensor([2., 2., 3.])
    """
    if isinstance(ord, Real) and np.isinf(ord):
        op = mg_max if ord > 0 else mg_min
        abs_ = absolute(x, constant=constant)
        out = op(abs_, axis=axis, keepdims=keepdims)

        in_ndim = abs_.creator.variables[0].ndim

        if (axis is None and ord is not None and in_ndim == 2) or (
            hasattr(axis, "__len__") and len(axis) > 1
        ):
            raise NotImplementedError(
                "mygrad.linalg.norm does not support matrix norms"
            )
        return out
    return Tensor._op(
        Norm,
        x,
        op_kwargs={
            "axis": axis,
            "keepdims": keepdims,
            "ord": ord,
            "nan_to_num": nan_to_num,
        },
        constant=constant,
    )


@implements_numpy_override()
def einsum(
    *operands: Union[ArrayLike, str, Sequence[int]],
    optimize: bool = False,
    out: Optional[Union[np.ndarray, Tensor]] = None,
    constant: Optional[bool] = None,
) -> Tensor:
    r"""
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

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant.grad`` will always return
        ``None``).

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

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
            operands[1:] = (
                var.data if isinstance(var, Tensor) else var for var in operands[1:]
            )
    else:
        # operands form: op0, sublist0, op1, sublist1, ..., [sublistout]
        end = -1 if len(operands) % 2 else None  # -1 if sublistout is included
        variables = operands[:end:2]
        if any(isinstance(i, Tensor) for i in operands):
            operands[:end:2] = (
                var.data if isinstance(var, Tensor) else var for var in operands[:end:2]
            )

    in_lbls, out_lbls, _ = _parse_einsum_input(operands)

    # einsum doesn't handle out=None properly in numpy 1.17

    return Tensor._op(
        EinSum,
        *variables,
        op_kwargs=dict(in_lbls=in_lbls, out_lbls=out_lbls, optimize=optimize),
        constant=constant,
        out=out,
    )
