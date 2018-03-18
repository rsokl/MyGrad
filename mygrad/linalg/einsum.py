from mygrad.operations.multivar_operations import MultiVarBroadcastableOp
from mygrad import Tensor
import numpy as np

from numpy.core.einsumfunc import _parse_einsum_input


class EinSum(MultiVarBroadcastableOp):
    def __call__(self, *variables, in_lbls, out_lbls, **kwargs):

        self.variables = variables
        return np.einsum("->".join((in_lbls, out_lbls)), *(var.data for var in self.variables))


def einsum(*operands, **kwargs):
    """ f(a, b, ...) -> a * b * ...

        Parameters
        ----------
        variables : Sequence[Union[tensor-like, Number]]

        Returns
        -------
        mygrad.Tensor"""


    in_lbls, out_lbls, variables = _parse_einsum_input(operands)
    return Tensor._op(EinSum, *variables, op_kwargs=dict(in_lbls=in_lbls,
                                                         out_lbls=out_lbls,
                                                         **kwargs))
