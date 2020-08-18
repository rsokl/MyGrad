import numpy as np

from mygrad.tensor_base import asarray


def check_loss_inputs(x, y_true):
    """ Ensures that the inputs to scores-truth style loss functions
    are of the correct shapes and types.

    Parameters
    ----------
    x : mygrad.Tensor, shape=(N, C)
        The C class scores for each of the N pieces of data.

    y_true : Sequence[int]
        The correct class-indices, in [0, C), for each datum.

    Raises
    ------
    TypeError
        `y_true` must be an integer-type array-like object
    ValueError
        `x` must be a 2-dimensional array-like object
        `y_true` must be a shape-(N,) array-like object
    """
    if not x.ndim == 2:
        raise ValueError(
            "`x` must be a 2-dimensional array-like object, got {}-dim".format(x.ndim)
        )

    y_true = asarray(y_true)
    if not np.issubdtype(y_true.dtype, np.integer):
        raise TypeError(
            "`y_true` must be an integer-type "
            "array-like object, got {}".format(y_true.dtype)
        )

    if y_true.ndim != 1 or y_true.shape[0] != x.shape[0]:
        raise ValueError(
            "`y_true` must be a shape-(N,) array: \n"
            "\tExpected shape-{}\n"
            "\tGot shape-{}".format((x.shape[0],), y_true.shape)
        )
