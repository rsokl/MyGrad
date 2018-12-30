from ...utils.numerical_gradient import numerical_gradient_full
from ...wrappers.sequence_func import fwdprop_test_factory, backprop_test_factory
from ...wrappers.uber import fwdprop_test_factory as uber_fwd
from ...wrappers.uber import backprop_test_factory as uber_bkwd
from ...custom_strategies import valid_axes
from numpy.testing import assert_allclose
from pytest import raises

from mygrad import amax, amin, sum, mean, cumprod, cumsum, prod, var, std
import mygrad

import numpy as np

from mygrad import Tensor
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import note

def axis_arg(*arrs): return valid_axes(arrs[0].ndim)

def keepdims_arg(*arrs): return st.booleans()

def ddof_arg(*arrs): return st.integers(0, min(arrs[0].shape) - 1)


@fwdprop_test_factory(mygrad_func=amax, true_func=np.amax)
def test_max_fwd(): pass


@uber_fwd(mygrad_func=amax, true_func=np.amax, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_max_fwd_uber(): pass

@uber_bkwd(mygrad_func=amax, true_func=np.amax, num_arrays=1,
           kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
           vary_each_element=True, index_to_unique={0: True},
           elements_strategy=st.integers)
def test_max_bkwd_uber(): pass

@backprop_test_factory(mygrad_func=amax, true_func=np.amax, unique=True)
def test_max_bkwd(): pass


@uber_fwd(mygrad_func=amin, true_func=np.amin, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_min_fwd(): pass


@uber_bkwd(mygrad_func=amin, true_func=np.amin, num_arrays=1,
           kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
           vary_each_element=True, index_to_unique={0: True},
           elements_strategy=st.integers)
def test_min_bkwd(): pass


def test_min_max_aliases():
    assert mygrad.max == amax
    assert mygrad.min == amin


@uber_fwd(mygrad_func=sum, true_func=np.sum, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_sum_fwd(): pass


@uber_bkwd(mygrad_func=sum, true_func=np.sum, num_arrays=1,
           kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
           vary_each_element=True)
def test_sum_bkwd(): pass


@uber_fwd(mygrad_func=mean, true_func=np.mean, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_mean_fwd(): pass


@uber_bkwd(mygrad_func=mean, true_func=np.mean, num_arrays=1,
           kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
           vary_each_element=True)
def test_mean_bkwd(): pass


@uber_fwd(mygrad_func=var, true_func=np.var, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                      ddof=ddof_arg))
def test_var_fwd(): pass


@uber_bkwd(mygrad_func=var, true_func=np.var, num_arrays=1,
           kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                       ddof=ddof_arg),
           vary_each_element=True, index_to_bnds={0: (-10, 10)},
           atol=1e-5, rtol=1e-5)
def test_var_bkwd(): pass


# std composes mygrad's sqrt and var, backprop need not be tested
@uber_fwd(mygrad_func=std, true_func=np.std, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                      ddof=ddof_arg))
def test_std_fwd(): pass


def _assume(*arrs, **kwargs):
    return all(i > 1 for i in arrs[0].shape)


@uber_bkwd(mygrad_func=std, true_func=np.std, num_arrays=1,
           kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                       ddof=ddof_arg),
           vary_each_element=True, index_to_bnds={0: (-10, 10)},
           elements_strategy=st.integers,
           index_to_unique={0: True},
           assumptions=_assume,
           atol=1e-5, rtol=1e-5)
def test_std_bkwd(): pass


@uber_fwd(mygrad_func=prod, true_func=np.prod, num_arrays=1,
          kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
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


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-2, 2), max_dims=4, max_side=5)
def test_cumprod_bkwd(): pass


@settings(deadline=None)
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


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-2, 2), max_dims=4, max_side=5)
def test_cumsum_bkwd(): pass

