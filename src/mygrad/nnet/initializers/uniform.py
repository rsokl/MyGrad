import numpy as np
from mygrad import Tensor


def uniform(*shape, lower_bound=0, upper_bound=1, dtype=np.float32, constant=False):
    """ Initialize a ``Tensor`` by drawing from a uniform distribution.

    Parameters
    ----------
    shape : Sequence[int]
        The output shape.

    lower_bound : Real, optional (default=0)
        Lower bound on the output interval, inclusive.

    upper_bound : Real, optional (default=1)
        Upper bound on the output interval, exclusive.

    dtype : data-type, optional (default=float32)
        The data type of the output tensor; must be a floating-point type.

    constant : bool, optional (default=False)
        If `True`, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor, shape=``shape``
        A Tensor, with values drawn uniformly from [lower_bound, upper_bound).
    """
    if lower_bound >= upper_bound:
        raise ValueError("Uniform lower bound cannot be greater than upper bound")
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("Uniform initialization requires a floating-point dtype")

    if len(shape) == 1:
        shape = shape[0]

    if isinstance(lower_bound, Tensor):
        lower_bound = lower_bound.item()
    if isinstance(upper_bound, Tensor):
        upper_bound = upper_bound.item()

    return Tensor(np.random.uniform(lower_bound, upper_bound, shape), dtype=dtype, constant=constant)
