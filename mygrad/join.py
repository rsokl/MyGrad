from .tensor_base import Tensor
from .operations import Concatenate, Dstack, Hstack, Stack, Vstack

__all__ = ["concatenate",
           "dstack",
           "hstack",
           "stack",
           "vstack"]


def concatenate(*variables, axis=0):
    """ Joins multiple Tensors along an existing axis.

        a -> [[1 2]
              [3 4]]
        b -> [[5 6]
              [7 8]]

        f(a, b, axis=1) -> [[1 2 5 6]
                            [3 4 7 8]]

        c -> [[[1]
               [2]
               [3]]]
        d -> [[[4]
               [5]
               [6]]]

        f(a, b, axis=0) -> [[[1]
                             [2]
                             [3]]

                            [[4]
                             [5]
                             [6]]]

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]
        axis : Optional[int] (default=0)
            The index of the axis in the resulting Tensor's dimensions along which
            the input variables will be concatenated.

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Concatenate, *variables, op_kwargs=dict(axis=axis))


def dstack(*variables):
    """ Stacks multiple Tensors depth-wise, along the 3rd axis.
        0-D, 1-D, and 2-D Tensors are first reshaped to (1,1,1), (1,N,1),
        and (N,M,1) Tensors, respectively.

        a -> [[1 2]
              [3 4]]
        b -> [[[-1 -2]
               [-3 -4]]

              [[-5 -6]
               [-7 -8]]]

        f(a, b) -> [[[1 -1 -2]
                     [2 -3 -4]]

                    [[3 -5 -6]
                     [4 -7 -8]]]

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Dstack, *variables)


def hstack(*variables):
    """ Stacks multiple Tensors horizontally, along the 2nd (column) axis.
        1-D Tensors are joined along the 1st axis. 0-D Tensors are first
        reshaped to (1,) Tensors.

        a -> [[1]
              [2]
              [3]]
        b -> [[4]
              [5]
              [6]]

        f(a, b) -> [[1 4]
                    [2 5]
                    [3 6]]

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Hstack, *variables)


def stack(*variables, axis=0):
    """ Stacks multiple Tensors along a new axis.

        a -> [1 2 3]
        b -> [4 5 6]

        f(a, b, axis=1) -> [[1 4]
                            [2 5]
                            [3 6]]

        c -> [[1 2]
              [3 4]]
        d -> [[5 6]
              [7 8]]

        f(a, b, axis=2) -> [[[1 5]
                             [2 6]]

                            [[3 7]
                             [4 8]]]

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]
        axis : Optional[int] (default=0)
            The index of the new axis in the resulting Tensor's dimensions. The
            input variables are stacked along this axis in the resulting Tensor.


        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Stack, *variables, op_kwargs=dict(axis=axis))


def vstack(*variables):
    """ Stacks multiple Tensors vertically, along the 1st (row) axis.
        0-D and 1-D Tensors are first reshaped into (1,1) and (1,N)
        Tensors, respectively.

        a -> [1 2 3]
        b -> [4 5 6]

        f(a, b) -> [[1 2 3]
                    [4 5 6]]

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(Vstack, *variables)
