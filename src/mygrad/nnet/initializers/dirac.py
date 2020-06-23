import numpy as np
from mygrad import Tensor


def dirac(*shape, dtype=np.float32, constant=False):
    """ Initialize a `Tensor` according to the Dirac initialization procedure described by Zagoruyko and Komodakis.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output Tensor. Note that `shape` must be at least two-dimensional.

    dtype : data-type, optional (default=float32)
        The data type of the output tensor.

    constant : bool, optional (default=False)
        If `True`, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor, shape=`shape`
        A Tensor, with values initialized according to the Dirac initialization.

    Extended Description
    --------------------
    Zagoruyko and Komodakis put forward the Dirac initialization in the paper
        "DiracNets: Training Very Deep Neural Networks without Skip Connections"
    https://arxiv.org/abs/1706.00388

    A Tensor I initialized via this should satisfy:
        I ⋆ x = x

    for compatible tensors x, where ⋆ indicates convolution. Note that this does not
    guarantee that the convolution will produce x, but it will preserve as many channels of
    the input as possible.
    """
    if len(shape) == 1:
        shape = shape[0]

    if len(shape) < 2:
        raise ValueError("Dirac initialization requires at least two dimensions")

    tensor = np.zeros(shape, dtype=dtype)
    minimum_depth = np.minimum(shape[0], shape[1])  # out dim, in dim
    depths = range(minimum_depth)
    trailing_indices = ([i // 2] * len(depths) for i in tensor.shape[2:])
    # tensor[i, i, k1//2, k2//2, ..., kn//2] for each i in min(shape[0], shape[1]
    # where the k values are the spatial dimensions of `tensor`
    tensor[(depths, depths, *trailing_indices)] = 1
    return Tensor(tensor, constant=constant)
