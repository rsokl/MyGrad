import numpy as np

from mygrad.nnet.initializers.uniform import uniform


def glorot_uniform(*shape, gain=1, dtype=np.float32, constant=False):
    r""" Initialize a `Tensor` according to the uniform initialization procedure
    described by Glorot and Bengio.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output Tensor. Note that `shape` must be at least two-dimensional.

    gain : Real, optional (default=1)
        The gain (scaling factor) to apply.

    dtype : data-type, optional (default=float32)
        The data type of the output tensor; must be a floating-point type.

    constant : bool, optional (default=False)
        If `True`, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values initialized according to the glorot uniform initialization.

    Extended Description
    --------------------
    Glorot and Bengio put forward this initialization in the paper
        "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        U[-\frac{\sqrt{6}}{\sqrt{n_j+n_{j+1}}}, \frac{\sqrt{6}}{\sqrt{n_j+n_{j+1}}}]
    """
    if len(shape) == 1:
        shape = shape[0]
    if len(shape) < 2:
        raise ValueError(
            "Glorot Uniform initialization requires at least two dimensions"
        )

    fan_in = shape[1] * (np.prod(shape[2:]) if len(shape) > 2 else 1)
    fan_out = shape[0] * (np.prod(shape[2:]) if len(shape) > 2 else 1)
    bound = gain * np.sqrt(6 / (fan_in + fan_out))
    return uniform(
        shape, lower_bound=-bound, upper_bound=bound, dtype=dtype, constant=constant
    )
