from .ops import Reshape, Squeeze
from mygrad.tensor_base import Tensor


__all__ = ["reshape", "squeeze"]


def reshape(a, newshape, constant=False):
    """ Returns a tensor with a new shape, without changing its data.

        Parameters
        ----------
        a : array_like
            The tensor to be reshaped

        newshape : Tuple[int, ...]
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D tensor of that length.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the tensor and remaining dimensions.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor
            ``a`` with its shape changed permuted.  A new tensor is returned.

        Notes
        -----
        ``reshape`` utilizes C-ordering, meaning that it reads & writes elements using
        C-like index ordering; the last axis index changing fastest, and, proceeding
        in reverse order, the first axis index changing slowest. """
    return Tensor._op(Reshape, a, op_args=(newshape,), constant=constant)


def squeeze(a, axis=None, constant=False):
    """
    Remove single-dimensional entries from the shape of a tensor.

    Parameters
    ----------
    a : array_like
        The tensor to be reshaped
    
    axis : Optional[int, Tuple[int, ...]]
        Selects a subset of the single-dimensional entries in the 
        shape. If an axis is selected with shape entry greater than 
        one, an error is raised.

    constant : bool, optional(default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor

    Raises
    ------
    ValueError
        If ``axis`` is not ``None``, and an axis being squeezed is not of length 1

    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.Tensor([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> mg.squeeze(x).shape
    (3,)
    >>> mg.squeeze(x, axis=0).shape
    (3, 1)
    >>> mg.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> mg.squeeze(x, axis=2).shape
    (1, 3)"""
    return Tensor._op(Squeeze, a, op_args=(axis,), constant=constant)