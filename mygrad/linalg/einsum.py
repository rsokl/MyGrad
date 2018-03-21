from mygrad.operations.multivar_operations import MultiVarBroadcastableOp
from mygrad import Tensor
import numpy as np
from itertools import chain

from numpy.core.einsumfunc import _parse_einsum_input


def reduce_broadcast(grad, outshape):

    if grad.shape == outshape:
        return grad

    if grad.ndim != len(outshape):
        assert grad.ndim > len(outshape)
        grad = grad.sum(axis=range(grad.ndim - len(outshape)))

    keepdims = tuple(n for n,i in enumerate(grad.shape) if i != outshape[n])
    if keepdims:
        grad = grad.sum(axis=keepdims, keepdims=True)
    return grad


class EinSum(MultiVarBroadcastableOp):
    def __call__(self, *variables, in_lbls, out_lbls, **kwargs):
        self.in_lbls = in_lbls
        self.out_lbls = out_lbls
        self.variables = variables
        return np.einsum("->".join((in_lbls, out_lbls)), *(var.data for var in self.variables))

    def backward_var(self, grad, index):
        """"""
        """
        example
        -------
        fwd:          "ijk, k -> ji", x, y
        bkwd (var: 0): "ji, k -> ijk", grad, y
        bkwd (var: 1): "ji, ijk -> k", grad, x
        """

        # ijk, k
        in_lbls = self.in_lbls.split(',')
        var_lbl = in_lbls.pop(index)

        # ji
        grad_lbl = self.out_lbls

        # catch indices over which uncontracted sum was performed
        # for the given variable: e.g for var-0 in "ijk, jk -> k"
        # i is summed over without contraction with another tensor
        unique_in_lbls = (set(chain.from_iterable(in_lbls)) | set(grad_lbl))

        if len(set(var_lbl) - unique_in_lbls) > 0:
            exp_dims = [slice(None) for i in range(grad.ndim)]
            grad_shape = list(grad.shape)
            for n, lbl in enumerate(var_lbl):
                if lbl not in unique_in_lbls:
                    grad_lbl = grad_lbl[:n] + lbl + grad_lbl[n:]
                    exp_dims.insert(n, np.newaxis)
                    grad_shape.insert(n, self.variables[index].shape[n])
            grad = np.broadcast_to(grad if not grad.ndim else grad[exp_dims], grad_shape)

        # ji, ijk -> k
        back_prop_lbls = ",".join([grad_lbl] + in_lbls) + "->" + var_lbl

        # grad, x
        arrays = tuple(i.data for i in self.variables)
        operands = (grad,) + arrays[:index] + arrays[index + 1:]

        # einsum(ji, ijk -> k", grad, x)
        outshape = self.variables[index].shape
        dfdx = reduce_broadcast(np.einsum(back_prop_lbls, *operands), outshape)
        self.variables[index].backward(dfdx)


def einsum(*operands, **kwargs):
    """ f(a, b, ...) -> a * b * ...

        Parameters
        ----------
        variables : Sequence[Union[ArrayLike, Real]]

        Returns
        -------
        mygrad.Tensor"""

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
