from ..utils.numerical_gradient import numerical_gradient_sequence
from ..custom_strategies import valid_axes

from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
from functools import wraps


from mygrad.linalg.einsum import einsum


def compare_einsum(*operands):
    mygrad_out = einsum(*operands)
    assert isinstance(mygrad_out, Tensor)
    assert np.allclose(np.einsum(*operands), einsum(*operands).data)


def test_einsum_fwd():
    a = np.arange(25).reshape(5, 5)
    b = np.arange(5)
    c = np.arange(6).reshape(2, 3)

    compare_einsum('ii', a)
    compare_einsum(a, [0, 0])

    compare_einsum('ii->i', a)
    compare_einsum(a, [0, 0], [0])

    compare_einsum('ij,j', a, b)
    compare_einsum(a, [0, 1], b, [1])

    compare_einsum('...j,j', a, b)

    compare_einsum('ji', c)
    compare_einsum(c, [1,0])

    compare_einsum('..., ...', 3, c)
    compare_einsum(3, [Ellipsis], c, [Ellipsis])

    compare_einsum('i,i', b, b)
    compare_einsum(b, [0], b, [0])

    compare_einsum('i,j', np.arange(2) + 1, b)
    compare_einsum('i...->...', a)

    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    compare_einsum('ijk,jil->kl', a, b)
    compare_einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3])

    a = np.arange(6).reshape((3, 2))
    b = np.arange(12).reshape((4, 3))
    compare_einsum('ki,jk->ij', a, b)
    compare_einsum(a, [0, 1], b, [2, 0], [1, 2])

    compare_einsum('ki,...k->i...', a, b)
    compare_einsum(a, [0, 1], b, [Ellipsis, 0], [1, Ellipsis])

    compare_einsum('k...,jk', a, b)
    compare_einsum(a, [0, Ellipsis], b, [2, 0])
