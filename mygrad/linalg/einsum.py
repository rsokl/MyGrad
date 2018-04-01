from mygrad.operation_base import BroadcastableOp
from mygrad import Tensor
from mygrad._utils import reduce_broadcast
import numpy as np
from itertools import chain
from functools import reduce

from numpy.core.einsumfunc import _parse_einsum_input
from numpy.lib.stride_tricks import as_strided

__all__ = ["einsum"]


def _unique_from_end(in_str):
    """ Return a string with all redundant characters removed,
        removing left-most redundant entries

        i.e. "ijikik" -> "jik"

        Parameters
        ----------
        in_str: str

        Returns
        -------
        str

        Examples
        --------
        >>> _unique_from_end("ijikik")
        "jik"
    """

    return reduce(lambda acc, x: acc + x if x not in acc else acc, in_str[::-1], '')[::-1]


def _merge_max_mappings(*mappings):
    """ Merge dictionaries based on largest values in key->value.

        Parameters
        ----------
        *mappings : Dict[Any, Any]

        Returns
        -------
        Dict[Any, Any]
        
        Examples
        --------
        >>> _merge_max_mappings({"a":1, "b":4}, {"a":2})
        {"a":2, "b":4}
    """

    def _merge_max(d1, d2):
        d1.update((k, v) for k, v in d2.items() if d1.get(k, 0) < v)
        return d1
    return reduce(_merge_max, mappings, {})


def _get_indices(item, seq):
    """ Return the indices where `item` occurs in `seq`

        Returns
        -------
        Generator[int]"""
    return (n for n, x in enumerate(seq) if x == item)


class EinSum(BroadcastableOp):
    def __call__(self, *variables, in_lbls, out_lbls, **kwargs):
        self.in_lbls = in_lbls
        self.out_lbls = out_lbls
        self.variables = variables
        return np.einsum("->".join((in_lbls, out_lbls)), *(var.data for var in self.variables), **kwargs)

    def backward_var(self, grad, index, **kwargs):
        """
        example
        -------
        fwd:          "ijk, k -> ji", x, y
        bkwd (var: 0): "ji, k -> ijk", grad, y
        bkwd (var: 1): "ji, ijk -> k", grad, x
        """

        numpy_arrays = tuple(i.data for i in self.variables)

        # ijk, k
        in_lbls = self.in_lbls.split(',')
        original_var_lbl = in_lbls.pop(index)
        var_lbl = _unique_from_end(original_var_lbl)
        repeat_lbls = len(var_lbl) != len(original_var_lbl)

        if repeat_lbls:
            # example fwd-prop: einsum("iji -> ij", x)
            # "iji" becomes "ji", later we will write along
            # the diagonal of an array to reinstate this axis that
            # we just removed
            mapping_gen = ({k: v for k, v in zip(lbl, arr.shape)}
                            for lbl, arr in zip(self.in_lbls.split(','), numpy_arrays))
            lbl_to_size = _merge_max_mappings(*mapping_gen)
            var_shape = tuple(lbl_to_size[lbl] for lbl in var_lbl)
        else:
            var_shape = self.variables[index].shape

        # ji
        grad_lbl = self.out_lbls

        # Catch indices over which un-contracted sum was performed
        # for the given variable: e.g for var-0 in "ijk, jk -> k"
        # i is summed over without contraction with another tensor
        #
        # Backpropping through this is illegal, as it requires the creation
        # of an axis; e.g. k, jk -> ijk
        # Broadcast the gradient along all such dimensions; e.g. k -> ik
        # then proceed as usual; e.g. ik, jk -> ijk
        unique_in_lbls = (set(chain.from_iterable(in_lbls)) | set(grad_lbl))
        if len(set(var_lbl) - unique_in_lbls) > 0:
            exp_dims = [slice(None) for i in range(grad.ndim)]
            grad_shape = list(grad.shape)
            for n, lbl in enumerate(var_lbl):
                if lbl not in unique_in_lbls:
                    grad_lbl = grad_lbl[:n] + lbl + grad_lbl[n:]
                    exp_dims.insert(n, np.newaxis)
                    grad_shape.insert(n, var_shape[n])

            grad = np.broadcast_to(grad if not grad.ndim else grad[exp_dims], grad_shape)

        # "ji, k -> ijk"
        back_prop_lbls = ",".join([grad_lbl] + in_lbls) + "->" + var_lbl

        # (grad, y)
        operands = (grad,) + numpy_arrays[:index] + numpy_arrays[index + 1:]

        if not repeat_lbls:
            # dfdx: einsum("ji, k -> ijk", grad, y)
            outshape = self.variables[index].shape
            dfdx = reduce_broadcast(np.einsum(back_prop_lbls, *operands), outshape)
            if var_shape != dfdx.shape:
                # if y was broadcast over x, the gradient needs to
                # be broadcast to x's shape: dfdx-shape (i,j,1) -> (i,j,k)
                dfdx = np.broadcast_to(dfdx, var_shape)
            self.variables[index].backward(dfdx, _broadcastable=False)
            return None

        # Accommodate trace by writing to strided view on array of zeros
        # For example:
        #
        # fwd:  einsum('ijkji, k -> jk', x, y)
        # dfdx: einsum('jk, k -> kji', grad, y, out=view_of_x)
        #
        # writing to `view_of_x`, which is a view along the appropriate
        # diagonals of x, is equivalent to:
        #
        # dfdx: einsum('jk, k -> ijkji', grad, y)
        #
        # which is formally correct but not supported by einsum.
        dfdx = np.zeros(tuple(lbl_to_size[i] for i in original_var_lbl))
        out_view_shape = tuple(lbl_to_size[i] for i in var_lbl)

        strides = tuple(sum(dfdx.strides[ind] for ind in _get_indices(lbl, original_var_lbl))
                        for lbl in var_lbl)
        out_view = as_strided(dfdx, shape=out_view_shape, strides=strides)
        np.einsum(back_prop_lbls, *operands, out=out_view)
        self.variables[index].backward(dfdx, **kwargs)


def einsum(*operands, **kwargs):
    """
    einsum(subscripts, *operands)

    The following docstring was adapted from the documentation for `numpy.einsum`

    Evaluates the Einstein summation convention on the operands.
    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion.  This function
    provides a way to compute such summations. The best way to understand this
    function is to try the examples below, which show how many common NumPy/MyGrad
    functions can be implemented as calls to `einsum`.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.

    operands : Tuple[ArrayLike, ...]
        The tensors used in the summation.

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
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
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
    of a new array.  Thus, taking the diagonal as ``einsum('ii->i', a)``
    produces a view.

    An alternative way to provide the subscripts and operands is as
    ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``. The examples
    below have corresponding `einsum` calls with the two parameter methods.

    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)
    
    >>> einsum('ii', a)
    Tensor(0)
    >>> einsum(a, [0,0])
    Tensor(60)
    >>> np.trace(a)
    Tensor(60)

    >>> einsum('ii->i', a)
    Tensor([ 0,  6, 12, 18, 24])
    >>> einsum(a, [0,0], [0])
    Tensor([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    >>> einsum('ij,j', a, b)
    Tensor([ 30,  80, 130, 180, 230])
    >>> einsum(a, [0,1], b, [1])
    Tensor([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> einsum('...j,j', a, b)
    Tensor([ 30,  80, 130, 180, 230])

    >>> einsum('ji', c)
    Tensor([[0, 3],
           [1, 4],
           [2, 5]])
    >>> einsum(c, [1,0])
    Tensor([[0, 3],
           [1, 4],
           [2, 5]])
    >>> c.T
    array([[0, 3],
           [1, 4],
           [2, 5]])

    >>> einsum('..., ...', 3, c)
    Tensor([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> einsum(',ij', 3, C)
    Tensor([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> einsum(3, [Ellipsis], c, [Ellipsis])
    Tensor([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    >>> einsum('i,i', b, b)
    Tensor(30)
    >>> einsum(b, [0], b, [0])
    Tensor(30)
    >>> np.inner(b,b)
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

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
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

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> einsum('ki,jk->ij', a, b)
    Tensor([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> einsum('ki,...k->i...', a, b)
    Tensor([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> einsum('k...,jk', a, b)
    Tensor([[10, 28, 46, 64],
           [13, 40, 67, 94]])

    >>> a = Tensor(np.zeros((3, 3)))
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
                                                         **kwargs))
