# -*- coding: utf-8 -*-
from mygrad import Tensor
import numpy as np


def rand(*shape):
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

    return Tensor(np.random.rand(*shape), constant=False)


def randn(*shape):
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

    return Tensor(np.random.randn(*shape), constant=False)


def randint(low, high=None, shape=None, dtype=int):
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

    return Tensor(np.random.randint(low, high, shape, dtype), constant=False)


def random_integers(low, high=None, shape=None):
    """ Random integers of type np.int_ between low and high, inclusive.


    Parameters
    ----------
    low: int
        Lowest (signed) integer to be drawn from the distribution 
        (unless high=None, in which case this parameter is the highest such integer).
    
    high: int, optional
        If provided, the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None).
    
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
        Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.    
    """

    return Tensor(np.random.random_integers(low, high, shape), constant=False)


def random_sample(shape=None):
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

    return Tensor(np.random.random_sample(shape), constant=False)


def random(shape=None):
    """ Return random floats in the half-open interval [0.0, 1.0). 
    Alias for random_sample to ease forward-porting to the new random API.
    Parameters
    
    To create a random sample of a given shape on the interval [a, b), call 
    (b-a) * random(shape) + a

    
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.    
    """

    return Tensor(np.random.random(shape), constant=False)


def ranf(shape=None):
    """ Return random floats in the half-open interval [0.0, 1.0). 
    Alias for random_sample to ease forward-porting to the new random API.
    Parameters
    
    To create a random sample of a given shape on the interval [a, b), call 
    (b-a) * ranf(shape) + a

    
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.    
    """

    return Tensor(np.random.ranf(shape), constant=False)


def sample(shape=None):
    """ Return random floats in the half-open interval [0.0, 1.0). 
    Alias for random_sample to ease forward-porting to the new random API.
    Parameters
    
    To create a random sample of a given shape on the interval [a, b), call 
    (b-a) * sample(shape) + a

    
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
    

    Returns
    -------
    int or mygrad.Tensor of ints
        size-shaped array of random integers from the appropriate distribution, or a single such random int if size not provided.    
    """

    return Tensor(np.random.sample(shape), constant=False)
