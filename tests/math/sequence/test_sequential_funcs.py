from ...utils.numerical_gradient import numerical_gradient_full
from ...wrappers.sequence_func import fwdprop_test_factory, backprop_test_factory
from numpy.testing import assert_allclose
from pytest import raises

from mygrad import amax, amin, sum, mean, cumprod, cumsum, prod, var, std
import mygrad

import numpy as np

from mygrad import Tensor
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

@fwdprop_test_factory(mygrad_func=amax, true_func=np.amax)
def test_max_fwd(): pass


@backprop_test_factory(mygrad_func=amax, true_func=np.amax, unique=True)
def test_max_bkwd(): pass


@fwdprop_test_factory(mygrad_func=amin, true_func=np.amin)
def test_min_fwd(): pass


@backprop_test_factory(mygrad_func=amin, true_func=np.amin, unique=True)
def test_min_bkwd(): pass


def test_min_max_aliases():
    assert mygrad.max == amax
    assert mygrad.min == amin


@fwdprop_test_factory(mygrad_func=sum, true_func=np.sum)
def test_sum_fwd(): pass


@backprop_test_factory(mygrad_func=sum, true_func=np.sum)
def test_sum_bkwd(): pass


@fwdprop_test_factory(mygrad_func=mean, true_func=np.mean)
def test_mean_fwd(): pass


@backprop_test_factory(mygrad_func=mean, true_func=np.mean)
def test_mean_bkwd(): pass


@fwdprop_test_factory(mygrad_func=var, true_func=np.var)
def test_var_fwd(): pass


@backprop_test_factory(mygrad_func=var, true_func=np.var)
def test_var_bkwd(): pass


@given(x=hnp.arrays(shape=(10,), dtype=float, elements=st.floats(-10, 10)),
       ddof=st.integers(0, 9))
def test_var_ddof(x, ddof):
    assert np.var(x, ddof=ddof) == var(x, ddof=ddof).data


@given(x=hnp.arrays(shape=(10,), dtype=float, elements=st.floats(-10, 10)),
       ddof=st.integers(0, 9))
def test_var_ddof_backward(x, ddof):
    y = Tensor(x)

    def f(z): return np.var(z, ddof=ddof)

    o = var(y, ddof=ddof)
    o.backward(2.)

    g, = numerical_gradient_full(f, x, back_grad=np.asarray(2.))
    assert_allclose(g, y.grad, rtol=1e-5, atol=1e-5)


# std composes mygrad's sqrt and var, backprop need not be tested
@fwdprop_test_factory(mygrad_func=std, true_func=np.std, unique=True)
def test_std_fwd(): pass


@given(x=hnp.arrays(shape=(10,), dtype=float, elements=st.floats(-10, 10)),
       ddof=st.integers(0, 9))
def test_std_ddof(x, ddof):
    assert np.std(x, ddof=ddof) == std(x, ddof=ddof).data


@fwdprop_test_factory(mygrad_func=prod, true_func=np.prod)
def test_prod_fwd(): pass


@backprop_test_factory(mygrad_func=prod, true_func=np.prod, xbnds=(-2, 2))
def test_prod_bkwd(): pass


def test_int_axis_cumprod():
    """check if numpy cumprod begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])
    with raises(TypeError):
        np.cumprod(x, axis=(0, 1))

    with raises(TypeError):
        cumprod(x, axis=(0, 1))


@fwdprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, single_axis_only=True, no_keepdims=True)
def test_cumprod_fwd(): pass


@settings(deadline=500)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-2, 2), max_dims=4, max_side=5)
def test_cumprod_bkwd(): pass


@settings(deadline=500)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-.5, .5), max_dims=4, max_side=5, unique=True,
                       draw_from_int=False)
def test_cumprod_bkwd2(): pass


def test_int_axis_cumsum():
    """check if numpy cumsum begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])
    with raises(TypeError):
        np.cumsum(x, axis=(0, 1))

    with raises(TypeError):
        cumsum(x, axis=(0, 1))


@fwdprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum, single_axis_only=True, no_keepdims=True)
def test_cumsum_fwd(): pass


@settings(deadline=500)
@backprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-2, 2), max_dims=4, max_side=5)
def test_cumsum_bkwd(): pass

