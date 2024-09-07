import numpy as np

from mygrad.tensor_base import Tensor


def glorot_normal(*shape, gain=1, dtype=np.float32, constant=None):
    r"""Initialize a :class:`mygrad.Tensor` according to the normal initialization procedure
    described by Glorot and Bengio.

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
        A Tensor, with values initialized according to the glorot normal initialization.

    Notes
    -----
    Glorot and Bengio put forward this initialization in the paper
        "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    A Tensor :math:`W` initialized in this way should be drawn from a distribution about

    .. math::
        \mathcal{N}(0, \frac{\sqrt{2}}{\sqrt{n_j+n_{j+1}}})
    """
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("Glorot Normal initialization requires a floating-point dtype")

    if len(shape) == 1:
        shape = shape[0]
    if len(shape) < 2:
        raise ValueError(
            "Glorot Normal initialization requires at least two dimensions"
        )

    if isinstance(gain, Tensor):
        gain = gain.item()

    fan_in = shape[1] * (shape[-1] if len(shape) > 2 else 1)
    fan_out = shape[0] * (shape[-1] if len(shape) > 2 else 1)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return Tensor(
        np.random.normal(0, std, shape),
        dtype=dtype,
        constant=constant,
        copy=False,
    )
