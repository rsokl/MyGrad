from mygrad.tensor_base import Tensor
from mygrad.math import power

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_power_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-3, 3)))
    a = Tensor(a)
    b = b

    result = a.data ** b

    o = a ** b
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)

    o = power(a, b)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_power_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))
    b = data.draw(hnp.arrays(shape=x.shape,
                             dtype=float,
                             elements=st.floats(-3, 3)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = a ** b
    c.backward(grad)
    assert np.allclose(a.grad, grad * b * (a.data ** (b - 1)))
