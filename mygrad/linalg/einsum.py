from mygrad.operations.multivar_operations import MultiVarBroadcastableOp
from mygrad import Tensor
import numpy as np

from numpy.core.einsumfunc import _parse_einsum_input


class EinSum(MultiVarBroadcastableOp):
    def __call__(self, *variables, in_lbls, out_lbls, **kwargs):
        self.in_lbls = in_lbls
        self.out_lbls = out_lbls
        self.variables = variables
        return np.einsum("->".join((in_lbls, out_lbls)), *(var.data for var in self.variables))

    def backward_var(self, grad, index):
        # fwd:          "ijk, k -> ji", x, y
        # bkwd (var-1): "ji, ijk -> k", x, grad

        # ijk, k
        in_lbls = self.in_lbls.split(',')

        # ji
        grad_lbl = self.out_lbls
        var_lbl = in_lbls[index]

        arrays = tuple(i.data for i in self.variables)

        # ji, ijk -> k
        back_prop_lbls = ",".join([grad_lbl] + in_lbls[:index] + in_lbls[index + 1:]) + "->" + var_lbl

        # grad, x
        operands = (grad,) + arrays[:index] + arrays[index + 1:]

        # einsum(ji, ijk -> k", x, grad)
        dfdx = np.einsum(back_prop_lbls, *operands)
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
