from mygrad.tensor_base import Tensor
from mygrad.math import sin, cos, tan, csc, sec, cot

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_sin_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.sin(a)

    o = sin(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_sin_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = sin(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad * np.cos(x))


@given(st.data())
def test_cos_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.cos(a)

    o = cos(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_cos_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = cos(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad * -np.sin(x))


@given(st.data())
def test_tan_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.tan(a)

    o = tan(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_tan_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = tan(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad / np.cos(x) ** 2)


@given(st.data())
def test_csc_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = 1 / np.sin(a)

    o = csc(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_csc_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = csc(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad * -np.cos(x) / np.sin(x) ** 2)


@given(st.data())
def test_sec_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = 1 / np.cos(a)

    o = sec(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_sec_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = sec(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad * np.sin(x) / np.cos(x) ** 2)


@given(st.data())
def test_cot_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = 1 / np.tan(a)

    o = cot(a)
    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_cot_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = cot(a)
    c.backward(grad)
    assert np.allclose(a.grad, grad / -np.sin(x) ** 2)
