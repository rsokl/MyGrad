import numpy as np

from mygrad import Tensor


def rand(*shape, constant=None):
    """Create a Tensor of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).

    Parameters
    ----------
    shape: d0, d1, ... dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        A ``shape``--shaped Tensor of floating-point samples from the uniform distribution
        over [0, 1), or a single such float if no parameters were supplied.

    Examples
    --------
    >>> from mygrad.random import rand
    >>> rand(3,4)
    Tensor([[0.9805903 , 0.82640985, 0.88230632, 0.73099815],
            [0.24845968, 0.12532893, 0.63171607, 0.32543228],
            [0.66029533, 0.79285341, 0.54967228, 0.25178508]])
    """

    return Tensor(np.random.rand(*shape), constant=constant, copy=False)


def randint(low, high=None, shape=None, dtype=int):
    """Return random integers from the “discrete uniform” distribution of the specified dtype in the
    “half-open” interval [low, high).

    If high is None (the default), then results are from [0, low).

    Parameters
    ----------
    low: int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution
        (unless high=None, in which case this parameter is one above the highest such integer).

    high: int or array-like of ints, optional
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if high=None). If array-like, must contain integer values

    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.

    dtype: dtype, optional
        Desired dtype of the result. Byteorder must be native. The default value is int.

    Returns
    -------
    int or mygrad.Tensor of ints
        ``shape``-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.

    Examples
    --------
    >>> from mygrad.random import randint
    >>> randint(low=1, high=7, shape=(2,5))
    Tensor([[2, 4, 1, 5, 1],
            [6, 2, 5, 4, 6]])

    >>> randint(low=4, high=100)
    Tensor(57)
    """

    return Tensor(np.random.randint(low, high, shape, dtype), copy=False)


def randn(*shape, constant=None):
    """Return a sample (or samples) from the “standard normal” distribution.

    Parameters
    ----------
    shape: shape: d0, d1, ... dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    mygrad.Tensor
        A ``shape``-shaped Tensor of floating-point samples from the standard normal distribution,
        or a single such float if no parameters were supplied.

    Examples
    --------
    >>> from mygrad.random import randn
    >>> randn(3, 3, 2)
    Tensor([[[-0.45664135,  0.05060159],
             [ 1.36883177, -0.46084292],
             [-0.76647664,  0.81667174]],

            [[ 0.08336453, -1.35104408],
             [ 0.73187355,  1.33405382],
             [ 0.28411209, -0.18047323]],

            [[-0.2239412 , -0.09170368],
             [-0.39175898,  0.81260396],
             [-1.28788909, -1.52525778]]])
    """

    return Tensor(np.random.randn(*shape), constant=constant, copy=False)


def random(shape=None, *, constant=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    To create a random sample of a given shape on the interval [a, b), call
    (b-a) * random(shape) + a

    Parameters
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    int or mygrad.Tensor of ints
        ``shape``-shaped array of random integers from the appropriate distribution, or a
        single such random int if size not provided.

    Examples
    --------
    >>> from mygrad.random import random
    >>> random((2, 4))
    Tensor([[0.14928578, 0.28812813, 0.56885892, 0.49555962],
            [0.19780163, 0.51162365, 0.7849505 , 0.47864586]])
    """

    return Tensor(np.random.random(shape), constant=constant, copy=False)


def random_sample(shape=None, *, constant=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    Results are from the “continuous uniform” distribution over the stated interval.

    To create a random sample of a given shape on the interval [a, b), call
    (b-a) * random_sample(shape) + a

    Parameters
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    int or mygrad.Tensor of ints
        ``shape``-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.

    Examples
    --------
    >>> from mygrad.random import random_sample
    >>> random_sample((3, 2))
    Tensor([[0.76490814, 0.69378441],
            [0.65228375, 0.68395309],
            [0.08228869, 0.03191064]])

    >>> random_sample()
    Tensor(0.47644928)
    """

    return Tensor(np.random.random_sample(shape), constant=constant, copy=False)


def ranf(shape=None, *, constant=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    To create a random sample of a given shape on the interval [a, b), call
    (b-a) * ranf(shape) + a

    Parameters
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    int or mygrad.Tensor of ints
        ``shape``-shaped array of random integers from the appropriate distribution, or
        a single such random int if size not provided.

    Examples
    --------
    >>> from mygrad.random import ranf
    >>> ranf((2, 3, 1))
    Tensor([[[0.9343681 ],
             [0.29573802],
             [0.84759669]],

            [[0.34563731],
             [0.68601617],
             [0.02388943]]])

    >>> ranf()
    Tensor(0.77739196)
    """

    return Tensor(np.random.ranf(shape), constant=constant, copy=False)


def sample(shape=None, *, constant=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    To create a random sample of a given shape on the interval [a, b), call
    (b-a) * sample(shape) + a

    Parameters
    ----------
    shape: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient)

    Returns
    -------
    int or mygrad.Tensor of ints
        ``shape``-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.

    Examples
    --------
    >>> from mygrad.random import sample
    >>> sample((3, 4))
    Tensor([[0.47263933, 0.10928814, 0.19737707, 0.30879006],
            [0.49870689, 0.05849937, 0.21095352, 0.09778017],
            [0.405788  , 0.91888808, 0.15061143, 0.63140668]])

    >>> sample()
    Tensor(0.50690423)
    """

    return Tensor(np.random.sample(shape), constant=constant, copy=False)


def seed(seed_number):
    """Seed the generator.

    Simply used NumPy's random state - i.e. this is equivalent to ``numpy.random.seed``.

    Parameters
    ----------
    seed_number : int or 1-d array_like, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers.

    Examples
    --------
    >>> from mygrad.random import seed, random
    >>> seed(0)
    >>> random((2, 4))
    Tensor([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
            [0.4236548 , 0.64589411, 0.43758721, 0.891773  ]])

    >>> seed(1)
    >>> random((2, 4))
    Tensor([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01],
            [1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01]]

    >>> seed(0)
    >>> random((2,4))
    Tensor([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
            [0.4236548 , 0.64589411, 0.43758721, 0.891773  ]])
    """

    np.random.seed(seed_number)
