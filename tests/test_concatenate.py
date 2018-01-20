from mygrad.tensor_base import Tensor
from mygrad.join import concatenate

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_concatenate_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))
    c = Tensor(a)
    b = b

    result = np.concatenate((a, b))
    o = concatenate(c, b)

    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


def test_concatenate_backward():
    a = Tensor([3])
    b = Tensor([4, 5, 6])
    c = concatenate(a, b)
    c.backward([-3, -4, -5, -6])

    assert a.data.shape == a.grad.shape
    assert np.allclose(a.data, np.abs(a.grad))

    assert b.data.shape == b.grad.shape
    assert np.allclose(b.data, np.abs(b.grad))


    d = Tensor([[1], [2]])
    e = Tensor([[3], [4]])
    f = concatenate(d, e, axis=1)
    f.backward([[-1, -3], [-2, -4]])

    assert d.data.shape == d.grad.shape
    assert np.allclose(d.data, np.abs(d.grad))

    assert e.data.shape == e.grad.shape
    assert np.allclose(e.data, np.abs(e.grad))


    g = Tensor([[[1, 2]], [[3, 4]]])
    h = Tensor([[[5, 6]], [[7, 8]]])
    i = concatenate(g, h, axis=2)
    i.backward([[[-1, -2, -5, -6]], [[-3, -4, -7, -8]]])

    assert g.data.shape == g.grad.shape
    assert np.allclose(g.data, np.abs(g.grad))

    assert h.data.shape == h.grad.shape
    assert np.allclose(h.data, np.abs(h.grad))
