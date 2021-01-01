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

    Examples
    --------
    >>> from mygrad.nnet.initializers import uniform
    >>> uniform(2, 3)
    Tensor([[0.8731087 , 0.30872548, 0.75528544],
            [0.55404514, 0.7652222 , 0.4955769 ]], dtype=float32)

    >>> uniform(2, 2, lower_bound=-1, upper_bound=3)
    Tensor([[ 1.9151938 , -0.28968155],
            [-0.01240687, -0.24448799]], dtype=float32)

    >>> uniform(5, dtype="float16", constant=True)
    Tensor([0.5186, 0.1481, 0.3745, 0.941 , 0.331 ], dtype=float16)
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

    return Tensor(
        np.random.uniform(lower_bound, upper_bound, shape),
        dtype=dtype,
        constant=constant,
        copy_data=False,
    )
