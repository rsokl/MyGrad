# -*- coding: utf-8 -*-
from mygrad import Tensor
import numpy as np


def random_sample(shape=None, constant=False):
    """ Return random floats in the half-open interval [0.0, 1.0).

    Results are from the “continuous uniform” distribution over the stated interval.
    
    To create a random sample of a given shape on the interval [a, b), call 
    (b-a) * random_sample(shape) + a


    Parameters
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.    
    """

    return Tensor(np.random.random_sample(shape), constant=constant)
