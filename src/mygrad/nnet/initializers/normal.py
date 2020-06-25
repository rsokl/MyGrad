import numpy as np
from mygrad import Tensor

def normal(*shape, mean=0, std=1, dtype=np.float32, constant=False):
    """ Initialize a :class:`mygrad.Tensor` by drawing from a normal (Gaussian) distribution.

    Parameters
    ----------
    shape : Sequence[int]
        The output shape.

    mean : Real, optional (default=0)
        The mean of the distribution from which to draw.

    std : Real, optional (default=1)
        The standard deviation of the distribution from which to draw.

    dtype : data-type, optional (default=float32)
        The data type of the output tensor; must be a floating-point type.

    constant : bool, optional (default=False)
        If `True`, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values drawn from Ɲ(μ, σ²), where μ=`mean` and σ=`std`.
    """
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("Glorot Normal initialization requires a floating-point dtype")
    if std < 0:
        raise ValueError("Standard deviation must be non-negative")

    if len(shape) == 1:
        shape = shape[0]

    if isinstance(mean, Tensor):
        mean = mean.item()
    if isinstance(std, Tensor):
        std = std.item()

    return Tensor(np.random.normal(mean, std, shape), dtype=dtype, constant=constant)
