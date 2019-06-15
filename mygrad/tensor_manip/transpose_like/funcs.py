from .ops import Transpose, MoveAxis, SwapAxes, Roll
from mygrad.tensor_base import Tensor

__all__ = ["transpose",
           "moveaxis",
           "swapaxes",
           "roll"]


def transpose(a, *axes, constant=False):
    """ Permute the dimensions of a tensor.

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
            `a` with its axes permuted.  A new tensor is returned.

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([[1, 2], [3, 4]])
        >>> a
        Tensor([[1, 2],
                [3, 4]])
        >>> a.transpose()
        Tensor([[1, 3],
                [2, 4]])
        >>> a.transpose((1, 0))
        Tensor([[1, 3],
                [2, 4]])
        >>> a.transpose(1, 0)
        Tensor([[1, 3],
                [2, 4]]) """
    if not axes:
        axes = None
    elif hasattr(axes[0], "__iter__") or axes[0] is None:
        if len(axes) > 1:
            raise TypeError("'{}' object cannot be interpreted as an integer".format(type(axes[0])))
        axes = axes[0]
    return Tensor._op(Transpose, a, op_args=(axes,), constant=constant)


def moveaxis(a, source, destination, constant=False):
    """ Move axes of a tensor to new positions. Other axes remain in their
        original order.


        Parameters
        ----------
        a : array_like
            The array whose axes should be reordered.

        source : Union[int, Sequence[int]]
            Original positions of the axes to move. These must be unique.

        destination : Union[int, Sequence[int]]
            Destination positions for each of the original axes. These must also be
            unique.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        result : mygrad.Tensor
            Array with moved axes. This array is a view of the input array..

        Examples
        --------
        >>> from mygrad import Tensor, moveaxis
        >>> x = Tensor(np.zeros((3, 4, 5)))
        >>> moveaxis(x, 0, -1).shape
        (4, 5, 3)
        >>> moveaxis(x, -1, 0).shape
        (5, 3, 4)
        >>> moveaxis(x, [0, 1], [-1, -2]).shape
        (5, 4, 3) """
    return Tensor._op(MoveAxis, a, op_args=(source, destination), constant=constant)


def swapaxes(a, axis1, axis2, constant=False):
    """ Interchange two axes of a tensor.

        Parameters
        ----------
        a : array_like
            Input array.

        axis1 : int
            First axis.

        axis2 : int
            Second axis.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor

        Examples
        --------
        >>> from mygrad import Tensor, swapaxes
        >>> x = Tensor([[1, 2, 3]])
        >>> swapaxes(x, 0, 1)
        Tensor([[1],
               [2],
               [3]])
        >>> x = Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        >>> x
        Tensor([[[0, 1],
                [2, 3]],
               [[4, 5],
                [6, 7]]])
        >>> swapaxes(x, 0, 2)
        Tensor([[[0, 4],
                [2, 6]],
               [[1, 5],
                [3, 7]]])
    """
    return Tensor._op(SwapAxes, a, op_args=(axis1, axis2), constant=constant)


def roll(a, shift, axis=None, constant=False):
    """
    Roll tensor elements along a given axis.

    Elements that roll beyond the end of an axis "wrap back around" to the beginning.

    This docstring was adapted from ``numpy.roll``

    Parameters
    ----------
    a : array_like
        Input tensor.

    shift : Union[int, Tuple[int, ...]]
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.

    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : Tensor
        Output array, with the same shape as `a`.


    Examples
    --------
    >>> import mygrad as mg
    >>> x = mg.arange(10)
    >>> mg.roll(x, 2)
    Tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> x2 = mg.reshape(x, (2,5))
    >>> x2
    Tensor([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> mg.roll(x2, 1)
    Tensor([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> mg.roll(x2, 1, axis=0)
    Tensor([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> mg.roll(x2, 1, axis=1)
    Tensor([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    """
    return Tensor._op(Roll, a, op_kwargs=dict(shift=shift, axis=axis), constant=constant)
