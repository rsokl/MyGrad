from mygrad.tensor_base import Tensor

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
    a = Tensor(a)
    b = b

    result = a.data * b
    assert np.allclose((a * b).data, result)
    assert np.allclose((b * a).data, result)



@given(st.data())
def test_add_backward(data):
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
    c = a * b
    c.backward(grad)
    assert np.allclose(a.grad, grad * b)

    a = Tensor(x)
    c = b * a
    c.backward(grad)
    assert np.allclose(a.grad, grad * b)


def test_broadcast():
    a = Tensor([3])
    b = Tensor([1, 2, 3])
    c = Tensor(2)
    f = a * b * c
    g = f.sum(keepdims=True)
    g.backward(-1)

    assert np.allclose(f.data, a.data*b.data*c.data)
    assert np.allclose(a.grad, np.array([-12]))
    assert np.allclose(b.grad, np.array([-6, -6, -6]))
    assert np.allclose(c.grad, np.array(-18))

