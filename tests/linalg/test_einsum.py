from ..utils.numerical_gradient import numerical_gradient_full
from ..custom_strategies import valid_axes, broadcastable_shape

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
    operands = tuple(i.data if isinstance(i, Tensor) else i for i in operands)
    assert np.allclose(np.einsum(*operands), einsum(*operands).data)


def backprop_linalg(f, *args, back_grad):
    grads = []
    args = tuple(i for i in args)
    for n in range(len(args)):
        tmp_f = lambda var: f(*args[:n], var, *args[n+1:])
        grads.append(numerical_gradient_full(tmp_f, x=args[n], back_grad=back_grad))
    return grads


def test_einsum_static_fwd():
    """ Check all einsum examples from numpy doc"""
    a = Tensor(np.arange(25).reshape(5, 5))
    b = Tensor(np.arange(5))
    c = Tensor(np.arange(6).reshape(2, 3))

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


@given(x=hnp.arrays(shape=(hnp.array_shapes(min_dims=1, max_dims=1)),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       data=st.data())
def test_einsum_bkwd1(x, data):
    x = Tensor(x)
    y = Tensor(data.draw(hnp.arrays(shape=x.shape,
                                    dtype=float,
                                    elements=st.floats(-100, 100))))

    grad = data.draw(st.floats(-100, 100))
    o = einsum("i, i", x, y)
    o.backward(grad)

    def f(x, y): return np.einsum("i, i", x, y)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)

    o.null_gradients()


def test_einsum_bkwd2():
    x = Tensor(np.random.rand(3, 4))
    y = Tensor(np.random.rand(3))
    grad = np.random.rand(4)

    o = einsum("ia, i -> a", x, y)
    o.backward(grad)

    def f(x, y): return np.einsum("ia, i -> a", x, y)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)

