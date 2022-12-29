import numpy as np

from mygrad.nnet.initializers.normal import normal


def he_normal(*shape, gain=1, dtype=np.float32, constant=None):
    r"""Initialize a :class:`mygrad.Tensor` according to the normal initialization procedure
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

    Notes
    -----
    He, Zhang, Ren, and Sun put forward this initialization in the paper
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification"
    https://arxiv.org/abs/1502.01852

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        \mathcal{N}(0, \sqrt{\frac{2}{(1+a^2)n_l}})

    where :math:`a` is the slope of the rectifier following this layer, which is incorporated
    using the `gain` variable above.

    The guidance put forward in that paper is that this initialization procedure should be preferred
    over the ``mygrad.nnet.initializers.glorot_*`` functions especially when rectifiers (e.g. ReLU,
    PReLU, leaky_relu) in very deep (> 1-20 or so layer) networks.

    Examples
    --------
    >>> from mygrad.nnet.initializers import he_normal
    >>> he_normal(2, 3)
    Tensor([[-2.3194842 ,  0.45956254, -0.28709933],
            [-0.15776408,  0.6777564 , -0.05587448]], dtype=float32)

    >>> he_normal(4, 2, gain=5/3, dtype="float64", constant=True)
    Tensor([[ 0.25962918,  1.1503933 ],
            [-0.13638746,  0.10581096],
            [ 1.44805926,  0.51367645],
            [-0.32018705, -0.80306442]])

    >>> he_normal(2, 1, 2, dtype="float16")
    Tensor([[[ 0.8057 , -0.2922 ]],
            [[ 0.12213, -0.715  ]]], dtype=float16)
    """
    if len(shape) == 1:
        shape = shape[0]
    if len(shape) < 2:
        raise ValueError("He Normal initialization requires at least two dimensions")

    std = gain / np.sqrt(shape[1] * (np.prod(shape[2:]) if len(shape) > 2 else 1))
    return normal(shape, mean=0, std=std, dtype=dtype, constant=constant)
