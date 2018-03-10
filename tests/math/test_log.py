from mygrad.tensor_base import Tensor
from mygrad.math import log, log2, log10
from ..custom_strategies import numerical_derivative

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np
import math


@given(st.data())
def test_log_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.log(a)

    o = log(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_log_backward(data):
    x, dx = data.draw(numerical_derivative(math.log, xbnds=[0, 100], no_go=(0,)))
    grad = data.draw(st.decimals(min_value=-100, max_value=100))
    var = Tensor(float(x))
    log(var).backward(float(grad))
    assert np.isclose(float(dx*grad), var.grad.item())


@given(st.data())
def test_log2_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.log2(a)

    o = log2(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_log2_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = log2(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad / (a.data * np.log(2)))


@given(st.data())
def test_log10_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.log10(a)

    o = log10(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_log10_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = log10(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad / (a.data * np.log(10)))
