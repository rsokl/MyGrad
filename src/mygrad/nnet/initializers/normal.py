import numpy as np

from mygrad import Tensor


def normal(*shape, mean=0, std=1, dtype=np.float32, constant=False):
    """Initialize a :class:`mygrad.Tensor` by drawing from a normal (Gaussian) distribution.

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

    Examples
    --------
    >>> from mygrad.nnet.initializers import normal
    >>> normal(1, 2, 3)
    Tensor([[[-0.06481607, -0.550582  ,  0.04689528],
             [ 0.82973075,  2.83742   ,  1.0964519 ]]], dtype=float32)

    >>> normal(2, 2, dtype="float16", constant=True)
    Tensor([[-1.335 ,  0.9297],
            [ 1.746 , -0.1222]], dtype=float16)

    >>> normal(5, dtype="float64")
    Tensor([-0.03875407,  0.65368466, -0.72636993,  1.57404148, -1.17444345])
    """
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("Normal initialization requires a floating-point dtype")
    if std < 0:
        raise ValueError("Standard deviation must be non-negative")

    if len(shape) == 1:
        shape = shape[0]

    if isinstance(mean, Tensor):
        mean = mean.item()
    if isinstance(std, Tensor):
        std = std.item()

    return Tensor(
        np.random.normal(mean, std, shape),
        dtype=dtype,
        constant=constant,
        copy=False,
    )
