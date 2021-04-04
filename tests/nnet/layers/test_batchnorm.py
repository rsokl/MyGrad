import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor
from mygrad.nnet.layers.batchnorm import batchnorm
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _mean(y, keepdims=False, axis=None, ddof=0):
    """For use in var"""
    if isinstance(axis, int):
        axis = (axis,)
    N = y.size if axis is None else np.prod([y.shape[i] for i in axis])
    return y.sum(keepdims=keepdims, axis=axis) / (N - ddof)


def _var(x, keepdims=False, axis=None, ddof=0):
    """Defines variance without using abs. Permits use of
    complex-step numerical derivative."""
    return _mean(
        (x - x.mean(axis=axis, keepdims=True)) ** 2,
        keepdims=keepdims,
        axis=axis,
        ddof=ddof,
    )


def simple_batchnorm(x, gamma=None, beta=None, eps=None):
    axes = [i for i in range(x.ndim)]
    axes.pop(1)  # every axis except 1
    axes = tuple(axes)
    keepdims_shape = tuple(1 if n != 1 else d for n, d in enumerate(x.shape))

    mean = mg.mean(x, axis=axes, keepdims=True)
    var = _var(x, axis=axes, keepdims=True)
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

    y1 = simple_batchnorm(t1, gamma=g1, beta=b1, eps=1e-10)
    y2 = batchnorm(t2, gamma=g2, beta=b2, eps=1e-10)

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

    assert not t2._ops

    if gamma is not None:
        assert not g2._ops

    if beta is not None:
        assert not b2._ops


@mg.no_autodiff
def simple_batchnorm_numpy(x, gamma=None, beta=None, eps=0):
    return mg.asarray(simple_batchnorm(x, eps=eps, gamma=gamma, beta=beta))


@settings(deadline=None)
@fwdprop_test_factory(
    mygrad_func=batchnorm,
    true_func=simple_batchnorm_numpy,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=4)},
    kwargs=lambda x: st.fixed_dictionaries(dict(eps=st.floats(1e-20, 1e0))),
    atol=1e-5,
)
def test_batchnorm_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=batchnorm,
    true_func=simple_batchnorm_numpy,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=4)},
    kwargs=lambda x: st.fixed_dictionaries(dict(eps=st.floats(1e-20, 1e0))),
    vary_each_element=True,
)
def test_batchnorm_bkwd():
    pass
