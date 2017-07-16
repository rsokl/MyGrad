from mygrad.tensor_base import Tensor
from mygrad.math import divide

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_add_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.01, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(0.01, 100)))
    result = a / b
    assert np.allclose((Tensor(a) / b).data, result)
    assert np.allclose((a / Tensor(b)).data, result)
    assert np.allclose(divide(Tensor(a), b).data, result)
    assert np.allclose(divide(a, Tensor(b)).data, result)


@given(st.data())
def test_add_backward(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.01, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(0.01, 100)))

    grad = data.draw(hnp.arrays(shape=a.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    x = Tensor(a)
    c = x / b
    c.backward(grad)
    assert np.allclose(x.grad, grad / b)

    x = Tensor(b)
    c = a / x
    c.backward(grad)
    assert np.allclose(x.grad,  grad * -a / x.data ** 2)

    x = Tensor(a)
    c = divide(x, b)
    c.backward(grad)
    assert np.allclose(x.grad, grad / b)

    x = Tensor(b)
    c = divide(a, x)
    c.backward(grad)
    assert np.allclose(x.grad, grad * -a / x.data ** 2)
