from mygrad.tensor_base import Tensor
from mygrad import stack

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_stack_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))
    c = Tensor(a)
    b = b

    result = np.stack((a, b))
    o = stack(c, b)

    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


def test_stack_backward():
    a = Tensor(3)
    b = Tensor(4)
    c = Tensor(5)
    d = stack(a, b, c)
    d.backward([-3, -4, -5])

    assert a.data.shape == a.grad.shape
    assert np.allclose(a.data, np.abs(a.grad))

    assert b.data.shape == b.grad.shape
    assert np.allclose(b.data, np.abs(b.grad))

    assert c.data.shape == c.grad.shape
    assert np.allclose(c.data, np.abs(c.grad))


    e = Tensor([1, 2])
    f = Tensor([3, 4])
    g = stack(e, f, axis=1)
    g.backward([[-1, -3], [-2, -4]])

    assert e.data.shape == e.grad.shape
    assert np.allclose(e.data, np.abs(e.grad))

    assert g.data.shape == g.grad.shape
    assert np.allclose(g.data, np.abs(g.grad))


    h = Tensor([[1], [2]])
    i = Tensor([[3], [4]])
    j = stack(h, i, axis=2)
    j.backward([[[-1, -3]], [[-2, -4]]])

    assert h.data.shape == h.grad.shape
    assert np.allclose(h.data, np.abs(h.grad))

    assert i.data.shape == i.grad.shape
    assert np.allclose(i.data, np.abs(i.grad))


    k = Tensor([[[1], [2]], [[3], [4]]])
    l = Tensor([[[5], [6]], [[7], [8]]])
    m = stack(k, l, axis=2)
    m.backward([[[[-1], [-5]], [[-2], [-6]]], [[[-3], [-7]], [[-4], [-8]]]])

    assert k.data.shape == k.grad.shape
    assert np.allclose(k.data, np.abs(k.grad))

    assert l.data.shape == l.grad.shape
    assert np.allclose(l.data, np.abs(l.grad))
