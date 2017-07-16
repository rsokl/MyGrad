from mygrad.tensor_base import Tensor

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


# @given(st.data())
# def test_add_fwd(data):
#     a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
#                              dtype=float,
#                              elements=st.floats(-100, 100)))
#     b = data.draw(hnp.arrays(shape=a.shape,
#                              dtype=float,
#                              elements=st.floats(-100, 100)))
#     result = a - b
#     assert np.allclose((Tensor(a) - b).data, result)
#     assert np.allclose((a - Tensor(b)).data, result)
#     assert np.allclose((Tensor(a) - Tensor(b)).data, result)
#
#
#
# @given(st.data())
# def test_add_backward(data):
#     a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
#                              dtype=float,
#                              elements=st.floats(-100, 100)))
#     b = data.draw(hnp.arrays(shape=a.shape,
#                              dtype=float,
#                              elements=st.floats(-100, 100)))
#
#     grad = data.draw(hnp.arrays(shape=a.shape,
#                                 dtype=float,
#                                 elements=st.floats(-100, 100)))
#
#     x = Tensor(a)
#     c = x - b
#     c.backward(grad)
#     assert np.allclose(x.grad, grad)
#
#     x = Tensor(b)
#     c = a - x
#     c.backward(grad)
#     assert np.allclose(x.grad, -1 * grad)
#
#     x = Tensor(a)
#     y = Tensor(b)
#     c = x - y
#     c.backward(grad)
#     assert np.allclose(x.grad, grad)
#     assert np.allclose(y.grad, -1 * grad)
#
#     x = Tensor(a)
#     c = subtract(x, b)
#     c.backward(grad)
#     assert np.allclose(x.grad, grad)
#
#     x = Tensor(b)
#     c = subtract(a, x)
#     c.backward(grad)
#     assert np.allclose(x.grad, -1 * grad)
#
#     x = Tensor(a)
#     y = Tensor(b)
#     c = subtract(x, y)
#     c.backward(grad)
#     assert np.allclose(x.grad, grad)
#     assert np.allclose(y.grad, -1 * grad)