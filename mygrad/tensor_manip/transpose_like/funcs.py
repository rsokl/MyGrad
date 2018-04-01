from .ops import Transpose
from mygrad.tensor_base import Tensor

__all__ = ["transpose"]

def transpose(a, axes=None):
    """ Permute the dimensions of an array.

        Parameters
        ----------
        a : array_like
            The tensor to be transposed

        axes : Optional[Tuple[int]]
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        Returns
        -------
        mygrad.Tensor
            `a` with its axes permuted.  A new tensor is returned. """
    return Tensor._op(Transpose, a, op_args=(axes,))
