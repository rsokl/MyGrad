from mygrad import Tensor
import mygrad as mg
from mygrad.nnet.layers.batchnorm import batchnorm
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp


def simple_batchnorm(x, gamma, beta, eps):
    axes = [i for i in range(x.ndim)]
    axes.pop(1)  # every axis except 1
    axes = tuple(axes)
    keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))

    mean = mg.mean(x, axis=axes, keepdims=True)
    var = mg.var(x, axis=axes, keepdims=True)
    gamma = gamma.reshape(keepdims_shape)
    beta = beta.reshape(keepdims_shape)
    return gamma * (x - mean) / mg.sqrt(var + eps) + beta


@given(x=hnp.arrays(shape=hnp.array_shapes(min_dims=2, max_dims=4),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       data=st.data())
def test_batchnorm(x, data):
    gamma = data.draw(hnp.arrays(shape=x.shape[1:2], dtype=float, elements=st.floats(-10, 10)), label="gamma")
    beta = data.draw(hnp.arrays(shape=x.shape[1:2], dtype=float, elements=st.floats(-10, 10)), label="beta")
    x_orig = np.copy(x)
    gamma_orig = np.copy(gamma)
    beta_orig = np.copy(beta)

    t1 = Tensor(x)
    t2 = Tensor(x)
    g1 = Tensor(gamma)
    g2 = Tensor(gamma)
    b1 = Tensor(beta)
    b2 = Tensor(beta)

    y1 = simple_batchnorm(t1, g1, b1, eps=1e-7)
    y2 = batchnorm(t2, gamma=g2, beta=b2, eps=1e-7)

    assert_allclose(actual=y2.data, desired=y1.data, atol=1e-7, rtol=1e-7)
    grad = data.draw(hnp.arrays(shape=y2.shape, dtype=t2.dtype, elements=st.floats(-10, 10)),
                     label='grad')
    grad_orig = np.copy(grad)
    
    y1.backward(grad)
    y2.backward(grad)

    assert_allclose(actual=t2.grad, desired=t1.grad, atol=1e-4, rtol=1e-4)
    assert_allclose(actual=b2.grad, desired=b1.grad, atol=1e-4, rtol=1e-4)
    assert_allclose(actual=g2.grad, desired=g1.grad, atol=1e-4, rtol=1e-4)

    for o, c in zip((x, gamma, beta, grad), (x_orig, gamma_orig, beta_orig, grad_orig)):
        assert_array_equal(o, c)

    assert not np.shares_memory(g2.grad, b2.grad)
    assert not np.shares_memory(grad, t2.grad)

    y2.null_gradients()
    assert t2.grad is None
    assert g2.grad is None
    assert b2.grad is None

