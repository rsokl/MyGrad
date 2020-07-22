from mygrad import Tensor
import numpy as np


def rand(*shape, constant=False):
    """ Create a Tensor of the given shape and populate it with random 
    samples from a uniform distribution over [0, 1).



    Parameters
    ----------
    shape: The dimensions of the returned array, must be non-negative. 
    If no argument is given a single Python float is returned.
    


    Returns
    -------
    mygrad.Tensor
    
    A (d0, d1, ..., dn)-shaped Tensor of floating-point samples from the uniform distribution over [0, 1), 
    or a single such float if no parameters were supplied.
    
    """

    return Tensor(np.random.rand(*shape), constant=constant)
