import numpy as np

from mygrad.nnet.initializers.uniform import uniform


def he_uniform(*shape, gain=1, dtype=np.float32, constant=False):
    r""" Initialize a ``mygrad.Tensor`` according to the uniform initialization procedure
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
        If `True`, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor, shape=``shape``
        A Tensor, with values initialized according to the He uniform initialization.

    Extended Description
    --------------------
    He, Zhang, Ren, and Sun put forward this initialization in the paper
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification"
    https://arxiv.org/abs/1502.01852

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        U[-\sqrt{\frac{6}{(1+a^2)n_l}}, \sqrt{\frac{6}{(1+a^2)n_l}}]

    where :math:`a` is the slope of the rectifier following this layer, which is incorporated
    using the `gain` variable above.

    The guidance put forward in that paper is that this initialization procedure should be prefered
    over the ``mygrad.nnet.initializers.glorot_*`` functions especially when rectifiers (e.g. ReLU,
    PReLU, leaky_relu) in very deep (> 1-20 or so layer) networks.

    Examples
    --------
    >>> from mygrad.nnet.initializers import he_uniform
    >>> he_uniform(2, 3)
    Tensor([[-0.97671795,  0.85518736, -0.8187388 ],
            [ 0.7599437 ,  0.94951814, -0.96755147]], dtype=float32)

    >>> he_uniform(4, 2, gain=5/3, dtype="float64", constant=True)
    Tensor([[-1.10372799, -0.16472136],
            [-1.32614867,  1.14142637],
            [ 0.78044471,  0.20562334],
            [-1.23968259,  1.0057054 ]])

    >>> he_uniform(2, 1, 2, dtype="float16")
    Tensor([[[-0.1233,  0.1023]],
            [[ 0.3845,  0.1003]]], dtype=float16)
    """
    if len(shape) == 1:
        shape = shape[0]
    if len(shape) < 2:
        raise ValueError("He Uniform initialization requires at least two dimensions")

    bound = gain / np.sqrt(3 / shape[1] * (np.prod(shape[2:]) if len(shape) > 2 else 1))
    return uniform(
        shape, lower_bound=-bound, upper_bound=bound, dtype=dtype, constant=constant
    )
