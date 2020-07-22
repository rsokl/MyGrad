# -*- coding: utf-8 -*-
from mygrad import Tensor
import numpy as np


def randint(low, high=None, shape=None, dtype=int, constant=False):
    """ Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high). 
    If high is None (the default), then results are from [0, low).

    Parameters
    ----------
    low: int or array-like of ints
    Lowest (signed) integers to be drawn from the distribution 
    (unless high=None, in which case this parameter is one above the highest such integer).
    
    high: int or array-like of ints, optional
    If provided, one above the largest (signed) integer to be drawn from the distribution 
    (see above for behavior if high=None). If array-like, must contain integer values
    
    shape: int or tuple of ints, optional
    Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
    Default is None, in which case a single value is returned.
    
    dtype: dtype, optional
    Desired dtype of the result. Byteorder must be native. The default value is int.

    Returns
    -------
    int or mygrad.Tensor of ints
    
    shape-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.
    
    """

    return Tensor(np.random.randint(low, high, shape, dtype), constant=constant)
