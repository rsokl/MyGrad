from .ops import Reshape
from mygrad.tensor_base import Tensor


__all__ = ["reshape"]


def reshape(a, newshape):
    """ Returns a tensor with a new shape, without changing its data.

        Parameters
        ----------
        a : array_like
            The tensor to be reshaped

        newshape : Tuple[int, ...]
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D array of that length.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.

        Returns
        -------
        mygrad.Tensor
            `a` with its shape changed permuted.  A new tensor is returned.

        Notes
        -----
        `reshape` utilizes C-ordering, meaning that it reads & writes elements using
        C-like index ordering; the last axis index changing fastest, and, proceeding
        in reverse order, the first axis index changing slowest. """
    return Tensor._op(Reshape, a, op_args=(newshape,))
