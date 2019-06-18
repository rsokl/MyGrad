import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor
from mygrad.nnet.layers.batchnorm import batchnorm


def simple_batchnorm(x, gamma, beta, eps):
    axes = [i for i in range(x.ndim)]
    axes.pop(1)  # every axis except 1
    axes = tuple(axes)
    keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))

    mean = mg.mean(x, axis=axes, keepdims=True)
    var = mg.var(x, axis=axes, keepdims=True)
    norm = (x - mean) / mg.sqrt(var + eps)

    if gamma is not None:
        gamma = gamma.reshape(keepdims_shape)
        norm *= gamma

    if beta is not None:
        beta = beta.reshape(keepdims_shape)
        norm += beta
    return norm


@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=2, max_dims=4),
        dtype=float,
        elements=st.floats(-100, 100),
    ),
    data=st.data(),
)
def test_batchnorm(x, data):
    # optionally draw affine parameters
    gamma = data.draw(
        st.none()
        | hnp.arrays(shape=x.shape[1:2], dtype=float, elements=st.floats(-10, 10)),
        label="gamma",
    )
    beta = data.draw(
        st.none()
        | hnp.arrays(shape=x.shape[1:2], dtype=float, elements=st.floats(-10, 10)),
        label="beta",
    )
    x_orig = np.copy(x)

    gamma_orig = np.copy(gamma) if gamma is not None else None
    beta_orig = np.copy(beta) if beta is not None else None

    t1 = Tensor(x)
    t2 = Tensor(x)

    g1 = Tensor(gamma) if gamma is not None else None
    g2 = Tensor(gamma) if gamma is not None else None

    b1 = Tensor(beta) if beta is not None else None
    b2 = Tensor(beta) if beta is not None else None

    y1 = simple_batchnorm(t1, g1, b1, eps=1e-6)
    y2 = batchnorm(t2, gamma=g2, beta=b2, eps=1e-6)

    assert_allclose(actual=y2.data, desired=y1.data, atol=1e-4, rtol=1e-4)
    grad = data.draw(
        hnp.arrays(shape=y2.shape, dtype=t2.dtype, elements=st.floats(-10, 10)),
        label="grad",
    )
    grad_orig = np.copy(grad)

    y1.backward(grad)
    y2.backward(grad)

    assert_allclose(actual=t2.grad, desired=t1.grad, atol=1e-4, rtol=1e-4)

    if beta is not None:
        assert_allclose(actual=b2.grad, desired=b1.grad, atol=1e-4, rtol=1e-4)
    else:
        assert b2 is None

    if gamma is not None:
        assert_allclose(actual=g2.grad, desired=g1.grad, atol=1e-4, rtol=1e-4)
    else:
        assert g2 is None

    for n, (o, c) in enumerate(
        zip((x, gamma, beta, grad), (x_orig, gamma_orig, beta_orig, grad_orig))
    ):
        if o is None or c is None:
            assert o is c, "('{x}', '{gamma}', '{beta}', '{grad}')[{n}]".format(
                x=x, gamma=gamma, beta=beta, grad=grad, n=n
            )
        else:
            assert_array_equal(
                o,
                c,
                err_msg="('{x}', '{gamma}', '{beta}', '{grad}')[{n}]".format(
                    x=x, gamma=gamma, beta=beta, grad=grad, n=n
                ),
            )

    if gamma is not None and beta is not None:
        assert not np.shares_memory(g2.grad, b2.grad)
    assert not np.shares_memory(grad, t2.grad)

    y2.null_gradients()
    assert t2.grad is None

    if gamma is not None:
        assert g2.grad is None

    if beta is not None:
        assert b2.grad is None
