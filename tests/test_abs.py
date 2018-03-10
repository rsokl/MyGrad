from mygrad.tensor_base import Tensor
from mygrad.math import abs

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_abs_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.abs(a)

    o = abs(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_abs_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = abs(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad * np.piecewise(a.data, [a.data < 0, a.data == 0, a.data > 0], [-1, 0, 1]))
