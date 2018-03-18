from mygrad.operations.multivar_operations import MultiVarBroadcastableOp
from mygrad import Tensor
import numpy as np

from numpy.core.einsumfunc import _parse_einsum_input


class EinSum(MultiVarBroadcastableOp):
    def __call__(self, *variables, in_lbls, out_lbls, **kwargs):

        self.variables = variables
        return np.einsum("->".join((in_lbls, out_lbls)), *(var.data for var in self.variables))

    def backward_var(self, grad, index):
        raise NotImplementedError


def einsum(*operands, **kwargs):
    """ f(a, b, ...) -> a * b * ...

        Parameters
        ----------
        variables : Sequence[Union[ArrayLike, Real]]

        Returns
        -------
        mygrad.Tensor"""

    operands = list(operands)
    if isinstance(operands[0], str):
        variables = operands[1:]
        operands[1:] = (var.data if isinstance(var, Tensor) else var for var in operands[1:])
    else:
        end = -1 if len(operands) % 2 else None
        variables = operands[:end:2]
        operands[:end:2] = (var.data if isinstance(var, Tensor) else var for var in operands[:end:2])

    in_lbls, out_lbls, _ = _parse_einsum_input(operands)
    return Tensor._op(EinSum, *variables, op_kwargs=dict(in_lbls=in_lbls,
                                                         out_lbls=out_lbls,
                                                         **kwargs))
