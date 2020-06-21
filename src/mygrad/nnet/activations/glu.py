from numpy import ndarray

from mygrad import multiply, Tensor
from .sigmoid import sigmoid


def glu(x, dim=-1, constant=False):
    """ Returns the Gated Linear Unit A * σ(B), where A and B are split from `x`.

    Parameters
    ----------
    x : mygrad.Tensor
        The input.

    dim : int, optional (default=-1)
        The dimension along which to split the input in half and apply the GLU.

    constant : boolean, optional (default=False)
        If ``True``, the returned tensor is a constant (it
        does not back-propagate a gradient).

    Returns
    -------
    mygrad.Tensor
        The result of applying the  Gated Linear Unit elementwise to the input.

    Extended Description
    --------------------
    The Gated Linear Unit was proposed in the paper
        "Language Modeling with Gated Convolutional Networks"
        Yann Dauphin, Angela Fan, Michael Auli, David Grangier
    available at https://arxiv.org/abs/1612.08083

    The GLU operation splits the input `x` in half along `dim`, storing the first half in A and the
    second in B. The return value is then A ⊙ σ(B), where ⊙ is elementwise multiplication and σ is
    the sigmoid function.

    Examples
    --------
    >>> import mygrad as mg
    >>> from mygrad.nnet.activations import glu
    >>> x = mg.arange(-5, 5)
    >>> x
    Tensor([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> y = glu(x); y
    Tensor([-2.5       , -2.92423431, -2.64239123, -1.90514825, -0.98201379])
    >>> y.backward()
    >>> x.grad
    array([ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0])
    """
    if isinstance(dim, (ndarray, Tensor)):
        dim = dim.item()

    if not isinstance(dim, int):
        raise TypeError(
            f"`dim` must be an integer-valued scalar, got {dim} (type {type(dim)})"
        )

    first_idx = list(slice(None) for _ in x.shape)
    second_idx = list(slice(None) for _ in x.shape)
    first_idx[dim] = slice(0, x.shape[dim] // 2)
    second_idx[dim] = slice(x.shape[dim] // 2, None)

    first_half = x[tuple(first_idx)]
    second_half = x[tuple(second_idx)]

    if first_half.shape != second_half.shape:
        raise ValueError(
            f"The shapes after splitting must be the same but got {first_half.shape} "
            "and {second_half.shape}"
        )
    return multiply(first_half, sigmoid(second_half), constant=constant)
