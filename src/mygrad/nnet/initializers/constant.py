from mygrad.tensor_creation.funcs import full


def constant(*shape, value=0.0, dtype=None, constant=None):
    """Initialize a :class:`mygrad.Tensor` of shape `shape` with a constant value.

    This function is a thin wrapper around ``mygrad.full``.

    Parameters
    ----------
    shape : Sequence[int]
        The output shape.

    value : Real, optional (default=0)
        The value with which to fill the tensor.

    dtype : data-type, optional (default=None)
        The data type of the output tensor, or None to match ``value``.

    constant : bool, optional (default=False)
        If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient).

    Returns
    -------
    Tensor
        A Tensor of ``value`` with the given shape and dtype.

    Examples
    ----------
    >>> import mygrad as mg
    >>> mg.nnet.initializers.constant(2, 3, value=1)
    Tensor([[1, 1, 1],
            [1, 1, 1]])

    >>> mg.nnet.initializers.constant((3, 3), value=7.1)
    Tensor([[7.1, 7.1, 7.1],
            [7.1, 7.1, 7.1],
            [7.1, 7.1, 7.1]])

    >>> mg.nnet.initializers.constant(4)
    Tensor([0., 0., 0., 0.])
    """
    if len(shape) == 1:
        shape = shape[0]

    return full(shape, value, dtype=dtype, constant=constant)
