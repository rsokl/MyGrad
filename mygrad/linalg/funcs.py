from .ops import *
from mygrad.tensor_base import Tensor

__all__ = ["matmul"]

def matmul(a, b, constant=False):
    """ ``f(a, b) -> matmul(a, b)``

        Parameters
        ----------
        a : array_like
        
        b : array_like

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor"""
    return Tensor._op(MatMul, a, b, constant=constant)