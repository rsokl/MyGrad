from mygrad.operations.multivar_operations import MultiVarBroadcastableOp
from mygrad import Tensor
import numpy as np
from itertools import chain

from numpy.core.einsumfunc import _parse_einsum_input
from numpy.lib.stride_tricks import as_strided
from itertools import filterfalse


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

    def seen(x, store=[]):
        seen = x in store
        if not seen:
            store.append(x)
        return seen

    return "".join((filterfalse(seen, in_str[::-1])))[::-1]


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
    assert len(mappings) > 0
    mapping = mappings[0]
    for mapp in mappings:
        for key, val in mapp.items():
            if mapping.get(key, 0) < val:
                mapping[key] = val
    return mapping


def _get_indices(item, seq):
    return (n for n, x in enumerate(seq) if x == item)


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

        numpy_arrays = tuple(i.data for i in self.variables)

        # ijk, k
        in_lbls = self.in_lbls.split(',')
        original_var_lbl = in_lbls.pop(index)
        var_lbl = _unique_from_end(original_var_lbl)
        repeat_lbls = len(var_lbl) != len(original_var_lbl)

        if repeat_lbls:
            # example fwd-prop: einsum("iji -> ij", x)
            # "iji" becomes "ji"
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

        # ji, ijk -> k
        back_prop_lbls = ",".join([grad_lbl] + in_lbls) + "->" + var_lbl

        # grad, x
        operands = (grad,) + numpy_arrays[:index] + numpy_arrays[index + 1:]

        if not repeat_lbls:
            # dfdx: einsum("ji, k -> ijk", grad, y)
            outshape = self.variables[index].shape
            dfdx = reduce_broadcast(np.einsum(back_prop_lbls, *operands), outshape)
            self.variables[index].backward(dfdx)
            return None

        # accommodate trace by writing to strided view on array of zeros
        # example
        # fwd:  einsum('ijkji, k -> jk', x, y)
        # dfdx: einsum('jk, k -> ijkji', grad, y)
        out = np.zeros(tuple(lbl_to_size[i] for i in original_var_lbl))
        out_view_shape = tuple(lbl_to_size[i] for i in var_lbl)

        strides = tuple(sum(out.strides[ind] for ind in _get_indices(lbl, original_var_lbl))
                        for lbl in var_lbl)
        out_view = as_strided(out, shape=out_view_shape, strides=strides)
        np.einsum(back_prop_lbls, *operands, out=out_view)

        dfdx = reduce_broadcast(out, self.variables[index].shape)
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
