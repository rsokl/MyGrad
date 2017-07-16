from mygrad.tensor_base import Tensor
from mygrad.math import subtract

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_add_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))
    result = a - b
    assert np.allclose((Tensor(a) - b).data, result)
    assert np.allclose((a - Tensor(b)).data, result)
    assert np.allclose((Tensor(a) - Tensor(b)).data, result)



@given(st.data())
def test_add_backward(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=a.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    x = Tensor(a)
    c = x - b
    c.backward(grad)
    assert np.allclose(x.grad, grad)

    x = Tensor(b)
    c = a - x
    assert isinstance(c, Tensor)
    c.backward(grad)
    assert np.allclose(x.grad, -1 * grad)

    x = Tensor(a)
    y = Tensor(b)
    c = x - y
    assert isinstance(c, Tensor)
    c.backward(grad)
    assert np.allclose(x.grad, grad)
    assert np.allclose(y.grad, -1 * grad)

    x = Tensor(a)
    c = subtract(x, b)
    assert isinstance(c, Tensor)
    c.backward(grad)
    assert np.allclose(x.grad, grad)

    x = Tensor(b)
    c = subtract(a, x)
    assert isinstance(c, Tensor)
    c.backward(grad)
    assert np.allclose(x.grad, -1 * grad)

    x = Tensor(a)
    y = Tensor(b)
    c = subtract(x, y)
    assert isinstance(c, Tensor)
    c.backward(grad)
    assert np.allclose(x.grad, grad)
    assert np.allclose(y.grad, -1 * grad)


def test_subtract_broadcast():
    a = Tensor([3])
    b = Tensor([1, 2, 3])
    c = Tensor(2)
    f = a - b - c
    g = f.sum(keepdims=True)
    g.backward()

    assert np.allclose(f.data, a.data - b.data - c.data)
    assert a.grad.shape == (1,)
    assert np.allclose(a.grad, np.array([3]))
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, np.array([-1, -1, -1]))
    assert c.grad.ndim == 0
    assert np.allclose(c.grad, np.array(-3))
