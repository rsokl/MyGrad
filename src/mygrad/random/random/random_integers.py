from mygrad import Tensor
import numpy as np


def random_integers(low, high=None, shape=None, constant=False):
    """ Random integers of type np.int_ between low and high, inclusive.

    Parameters
    ----------
    low: int
        Lowest (signed) integer to be drawn from the distribution 
        (unless high=None, in which case this parameter is the highest such integer).
    
    high: int, optional
        If provided, the largest (siagned) integer to be drawn from the distribution (see above for behavior if high=None).
    
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
        Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.    
    """

    return Tensor(np.random.random_integers(low, high, shape), constant=constant)
