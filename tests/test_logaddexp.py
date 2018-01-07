from mygrad.tensor_base import Tensor
from mygrad.math import logaddexp

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_logaddexp_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    result = np.logaddexp(a, b)
    assert np.allclose(logaddexp(Tensor(a), b).data, result)
    assert np.allclose(logaddexp(a, Tensor(b)).data, result)


@given(st.data())
def test_logaddexp_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    a = Tensor(x)
    c = logaddexp(a, b)
    c.backward(grad)
    assert np.allclose(a.grad, grad * x / (np.exp(x) + np.exp(b)))

    a = Tensor(x)
    c = logaddexp(b, a)
    c.backward(grad)
    assert np.allclose(a.grad, grad * x / (np.exp(x) + np.exp(b)))


def test_logaddexp_broadcast():
    a = Tensor([3])
    b = Tensor([1, 2, 3])
    f = logaddexp(a, b)
    g = f.sum(keepdims=False)
    g.backward(-1)

    assert np.allclose(f.data, np.logaddexp(a.data, b.data))
    assert a.grad.shape == (1,)
    assert np.allclose(a.grad, -1 * np.array([sum(map(lambda x: 3 / (np.exp(3) + np.exp(x)), b.data))]))
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, -1 * np.array(list(map(lambda x: x / (np.exp(3) + np.exp(x)), b.data))))


    b = Tensor([1, 2, 3])
    c = Tensor(2)
    f = logaddexp(b, c)
    g = f.sum(keepdims=False)
    g.backward(-1)

    assert np.allclose(f.data, np.logaddexp(b.data, c.data))
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, -1 * np.array(list(map(lambda x: x / (np.exp(2) + np.exp(x)), b.data))))
    assert c.grad.ndim == 0
    assert np.allclose(c.grad, -1 * np.array(sum(map(lambda x: 2 / (np.exp(2) + np.exp(x)), b.data))))
