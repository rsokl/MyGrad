from mygrad.tensor_base import Tensor
from mygrad.math import logaddexp, log, exp

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
    b = Tensor(b)
    logaddexp(a, b).backward(grad)

    a1 = Tensor(a)
    b1 = Tensor(b)
    log(exp(a1) + exp(b1)).backward(grad)

    assert np.allclose(a.grad, a1.grad)
    assert np.allclose(b.grad, b1.grad)


def test_logaddexp_broadcast():
    a = Tensor([3])
    b = Tensor([1, 2, 3])
    out = logaddexp(a, b)
    out.sum(keepdims=False).backward(-1)

    a1 = Tensor(a)
    b1 = Tensor(b)
    long_form = log(exp(a1) + exp(b1))
    long_form.sum(keepdims=False).backward(-1)

    assert np.allclose(out.data, np.logaddexp(a.data, b.data))
    assert a.grad.shape == (1,)
    assert np.allclose(a.grad, a1.grad)
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, b1.grad)

    a = Tensor([1, 2, 3])
    b = Tensor(2)

    out = logaddexp(a, b)
    out.sum(keepdims=False).backward(-1)

    a1 = Tensor(a)
    b1 = Tensor(b)
    long_form = log(exp(a1) + exp(b1))
    long_form.sum(keepdims=False).backward(-1)

    assert np.allclose(out.data, np.logaddexp(a.data, b.data))
    assert a.grad.shape == (3,)
    assert np.allclose(a.grad, a1.grad)
    assert b.grad.ndim == 0
    assert np.allclose(b.grad, b1.grad)


