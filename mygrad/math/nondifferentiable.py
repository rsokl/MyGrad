import numpy as np

__all__ = ["argmin",
           "argmax"]

def argmax(a, axis=None, out=None):
    """ Returns the indices of the maximum values along an axis.

        Parameters
        ----------
        a: mygrad.Tensor
        
        axis: int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.
        
        out: numpy.array, optional
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
        
        Returns
        -------
        numpy.ndarray[int]"""
    
    return np.argmax(a.data, axis, out)
    
def argmin(a, axis=None, out=None):
    """ Returns the indices of the minimum values along an axis.

        Parameters
        ----------
        a: mygrad.Tensor
        
        axis: int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.
        
        out: numpy.array, optional
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
        
        Returns
        -------
        numpy.ndarray[int]"""
    
    return np.argmin(a.data, axis, out)