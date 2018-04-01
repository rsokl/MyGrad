from mygrad.tensor_base import Tensor
from mygrad import arcsin, arccos, arctan, arccsc, arcsec, arccot

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np
from numpy.testing import assert_allclose

@given(st.data())
def test_arcsin_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-0.9, 0.9)))

    result = np.arcsin(a)

    o = arcsin(a)
    assert isinstance(o, Tensor)
    assert_allclose(o.data, result)


@given(st.data())
def test_arcsin_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-0.9, 0.9)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = arcsin(a)
    c.backward(grad)
    assert_allclose(a.grad, np.select([np.abs(x) != 1], [grad / np.sqrt(1 - x ** 2)]))


@given(st.data())
def test_arccos_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-0.9, 0.9)))

    result = np.arccos(a)

    o = arccos(a)
    assert isinstance(o, Tensor)
    assert_allclose(o.data, result)


@given(st.data())
def test_arccos_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-0.9, 0.9)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = arccos(a)
    c.backward(grad)
    assert_allclose(a.grad, np.select([np.abs(x) != 1], [-grad / np.sqrt(1 - x ** 2)]))


@given(st.data())
def test_arctan_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.arctan(a)

    o = arctan(a)
    assert isinstance(o, Tensor)
    assert_allclose(o.data, result)


@given(st.data())
def test_arctan_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = arctan(a)
    c.backward(grad)
    assert_allclose(a.grad, grad / (1 + x ** 2))


@given(st.data())
def test_arccsc_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, -1.1)))

    result = np.arcsin(1 / a)

    o = arccsc(a)
    assert isinstance(o, Tensor)
    assert_allclose(o.data, result)


@given(st.data())
def test_arccsc_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(1.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = arccsc(a)
    c.backward(grad)
    assert_allclose(a.grad, np.select([np.abs(x) != 1], [-grad / (np.abs(x) * np.sqrt(x ** 2 - 1))]))


@given(st.data())
def test_arcsec_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, -1.1)))

    result = np.arccos(1 / a)

    o = arcsec(a)
    assert isinstance(o, Tensor)
    assert_allclose(o.data, result)


@given(st.data())
def test_arcsec_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(1.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = arcsec(a)
    c.backward(grad)
    assert_allclose(a.grad, np.select([np.abs(x) != 1], [grad / (np.abs(x) * np.sqrt(x ** 2 - 1))]))


@given(st.data())
def test_arccot_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 10)))

    result = np.arctan(1 / a)

    o = arccot(a)
    assert isinstance(o, Tensor)
    assert_allclose(o.data, result)


@given(st.data())
def test_arccot_backward(data):
    x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(0.1, 100)))

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(-10, 10)))

    a = Tensor(x)
    c = arccot(a)
    c.backward(grad)
    assert_allclose(a.grad, -grad / (1 + x ** 2))
