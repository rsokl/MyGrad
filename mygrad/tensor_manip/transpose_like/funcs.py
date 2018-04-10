from .ops import Transpose, MoveAxis, SwapAxes
from mygrad.tensor_base import Tensor

__all__ = ["transpose",
           "moveaxis",
           "swapaxes"]


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


def moveaxis(a, source, destination):
    """ Move axes of an array to new positions. Other axes remain in their
        original order.


        Parameters
        ----------
        a : array_like
            The array whose axes should be reordered.

        source : int or sequence of int
            Original positions of the axes to move. These must be unique.

        destination : int or sequence of int
            Destination positions for each of the original axes. These must also be
            unique.

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
    return Tensor._op(MoveAxis, a, op_args=(source, destination))


def swapaxes(a, axis1, axis2):
    """ Interchange two axes of a tensor.

        Parameters
        ----------
        a : array_like
            Input array.

        axis1 : int
            First axis.

        axis2 : int
            Second axis.

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
    return Tensor._op(SwapAxes, a, op_args=(axis1, axis2))
