# -*- coding: utf-8 -*-
from mygrad import Tensor
import numpy as np


def randn(*shape, constant=False):
    """ Return a sample (or samples) from the “standard normal” distribution.



    Parameters
    ----------
    shape: The dimensions of the returned array, must be non-negative. 
    If no argument is given a single Python float is returned.
    


    Returns
    -------
    mygrad.Tensor
    
    A (d0, d1, ..., dn)-shaped Tensor of floating-point samples from the standard normal distribution, 
    or a single such float if no parameters were supplied.
    
    """

    return Tensor(np.random.randn(*shape), constant=constant)
