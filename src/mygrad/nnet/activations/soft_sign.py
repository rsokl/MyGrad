from mygrad import abs, divide

__all__ = ["soft_sign"]


def soft_sign(x, constant=False):
    """ Returns the soft sign function x / (1 + |x|).

    Parameters
    ----------
    x : mygrad.Tensor
        Input data.

    constant : boolean, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor
        The soft sign function applied to `x` elementwise.
    """
    return divide(x, 1 + abs(x), constant=constant)
