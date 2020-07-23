from mygrad import Tensor
import numpy as np


def sample(shape=None, constant=False):
    """ Return random floats in the half-open interval [0.0, 1.0). 
    Alias for random_sample to ease forward-porting to the new random API.
    Parameters
    
    To create a random sample of a given shape on the interval [a, b), call 
    (b-a) * sample(shape) + a

    Parameters
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution,
         or a single such random int if size not provided.    
    """

    return Tensor(np.random.sample(shape), constant=constant)
