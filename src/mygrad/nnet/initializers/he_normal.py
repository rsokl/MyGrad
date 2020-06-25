import numpy as np
from mygrad import Tensor


def he_normal(*shape, gain=1, dtype=np.float32, constant=False):
    """ Initialize a ``Tensor`` according to the normal initialization procedure
    described by He et al.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output Tensor. Note that ``shape`` must be at least two-dimensional.

    gain : Real, optional (default=1)
        The gain (scaling factor) to apply.

    dtype : data-type, optional (default=float32)
        The data type of the output tensor; must be a floating-point type.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor, shape=``shape``
        A Tensor, with values initialized according to the He normal initialization.

    Extended Description
    --------------------
    He, Zhang, Ren, and Sun put forward this initialization in the paper
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification"
    https://arxiv.org/abs/1502.01852

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        \mathcal{N}(0, \sqrt{\frac{2}{(1+a^2)n_l}})

    where :math:`a` is the slope of the rectifier following this layer, which is incorporated
    using the `gain` variable above.
    """
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("Glorot Normal initialization requires a floating-point dtype")

    if len(shape) == 1:
        shape = shape[0]
    if len(shape) < 2:
        raise ValueError("He Normal initialization requires at least two dimensions")

    std = gain / np.sqrt(shape[1] * (np.prod(shape[2:]) if len(shape) > 2 else 1))
    return Tensor(np.random.normal(0, std, shape), dtype=dtype, constant=constant)
