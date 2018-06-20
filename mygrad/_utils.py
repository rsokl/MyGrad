def reduce_broadcast(grad, var_shape):
    """ Sum-reduce axes of `grad` so its shape matches `var_shape.

        This the appropriate mechanism for backpropagating a gradient
        through an operation in which broadcasting occurred for the
        given variable.

        Parameters
        ----------
        grad : numpy.ndarray
        var_shape : Tuple[int, ...]

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            The dimensionality of the gradient cannot be less than
            that of its associated variable."""
    if grad.shape == var_shape:
        return grad

    if grad.ndim != len(var_shape):
        if grad.ndim < len(var_shape):
            raise ValueError("The dimensionality of the gradient of the broadcasted operation"
                             "is less than that of its associated variable")
        grad = grad.sum(axis=tuple(range(grad.ndim - len(var_shape))))

    keepdims = tuple(n for n, i in enumerate(grad.shape) if i != var_shape[n])
    if keepdims:
        grad = grad.sum(axis=keepdims, keepdims=True)

    return grad