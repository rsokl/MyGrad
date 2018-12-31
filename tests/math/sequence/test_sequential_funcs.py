from ...wrappers.uber import fwdprop_test_factory as fwdprop_test_factory
from ...wrappers.uber import backprop_test_factory as backprop_test_factory
from ...custom_strategies import valid_axes
from pytest import raises

from mygrad import amax, amin, sum, mean, cumprod, cumsum, prod, var, std
import mygrad

import numpy as np

from hypothesis import settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


def axis_arg(*arrs):
    """ Wrapper for passing valid-axis search strategy to test factory"""
    return valid_axes(arrs[0].ndim)


def single_axis_arg(*arrs):
    """ Wrapper for passing valid-axis (single-value only)
    search strategy to test factory"""
    return valid_axes(arrs[0].ndim, single_axis_only=True)


def keepdims_arg(*arrs):
    """ Wrapper for passing keep-dims strategy to test factory"""
    return st.booleans()


def ddof_arg(*arrs):
    """ Wrapper for passing ddof strategy to test factory
    (argument for var and std)"""
    return st.integers(0, min(arrs[0].shape) - 1)


@fwdprop_test_factory(mygrad_func=amax, true_func=np.amax, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_max_fwd(): pass


@backprop_test_factory(mygrad_func=amax, true_func=np.amax, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, index_to_unique={0: True},
                       elements_strategy=st.integers)
def test_max_bkwd(): pass


@fwdprop_test_factory(mygrad_func=amin, true_func=np.amin, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_min_fwd(): pass


@backprop_test_factory(mygrad_func=amin, true_func=np.amin, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, index_to_unique={0: True},
                       elements_strategy=st.integers)
def test_min_bkwd(): pass


def test_min_max_aliases():
    assert mygrad.max == amax
    assert mygrad.min == amin


@fwdprop_test_factory(mygrad_func=sum, true_func=np.sum, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_sum_fwd(): pass


@backprop_test_factory(mygrad_func=sum, true_func=np.sum, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, atol=1e-5)
def test_sum_bkwd(): pass


@fwdprop_test_factory(mygrad_func=mean, true_func=np.mean, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_mean_fwd(): pass


@backprop_test_factory(mygrad_func=mean, true_func=np.mean, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       index_to_bnds={0: (-10, 10)},
                       vary_each_element=True)
def test_mean_bkwd(): pass


@fwdprop_test_factory(mygrad_func=var, true_func=np.var, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                                  ddof=ddof_arg))
def test_var_fwd(): pass


@backprop_test_factory(mygrad_func=var, true_func=np.var, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                                   ddof=ddof_arg),
                       vary_each_element=True, index_to_bnds={0: (-10, 10)},
                       atol=1e-5, rtol=1e-5)
def test_var_bkwd(): pass


# std composes mygrad's sqrt and var, backprop need not be tested
@fwdprop_test_factory(mygrad_func=std, true_func=np.std, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                                  ddof=ddof_arg))
def test_std_fwd(): pass


def _assume(*arrs, **kwargs):
    return all(i > 1 for i in arrs[0].shape)


@backprop_test_factory(mygrad_func=std, true_func=np.std, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                                   ddof=ddof_arg),
                       vary_each_element=True, index_to_bnds={0: (-10, 10)},
                       elements_strategy=st.integers,
                       index_to_unique={0: True},
                       assumptions=_assume,
                       atol=1e-5, rtol=1e-5)
def test_std_bkwd(): pass


@fwdprop_test_factory(mygrad_func=prod, true_func=np.prod, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_prod_fwd(): pass


@backprop_test_factory(mygrad_func=prod, true_func=np.prod, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, index_to_bnds={0: (-2, 2)})
def test_prod_bkwd(): pass


def test_int_axis_cumprod():
    """check if numpy cumprod begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])
    with raises(TypeError, message="`np.cumprod` is expected to raise a TypeError "
                                   "when it is provided a tuple of axes."):
        np.cumprod(x, axis=(0, 1))

    with raises(TypeError, message="`mygrad.cumprod` is expected to raise a TypeError "
                                   "when it is provided a tuple of axes."):
        cumprod(x, axis=(0, 1))


@fwdprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, num_arrays=1,
                      kwargs=dict(axis=single_axis_arg))
def test_cumprod_fwd(): pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, num_arrays=1,
                       kwargs=dict(axis=single_axis_arg),
                       vary_each_element=True, index_to_bnds={0: (-2, 2)})
def test_cumprod_bkwd(): pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, num_arrays=1,
                       kwargs=dict(axis=single_axis_arg),
                       vary_each_element=True, index_to_bnds={0: (-.5, .5)},
                       index_to_unique={0: True},
                       index_to_arr_shapes={0: hnp.array_shapes(max_side=5, max_dims=4)})
def test_cumprod_bkwd2(): pass


def test_int_axis_cumsum():
    """check if numpy cumsum begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])
    with raises(TypeError, message="`np.cumsum` is expected to raise a TypeError "
                                   "when it is provided a tuple of axes."):
        np.cumsum(x, axis=(0, 1))

    with raises(TypeError, message="`mygrad.cumsum` is expected to raise a TypeError "
                                   "when it is provided a tuple of axes."):
        cumsum(x, axis=(0, 1))


@fwdprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum, num_arrays=1,
                      kwargs=dict(axis=single_axis_arg))
def test_cumsum_fwd(): pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum, num_arrays=1,
                       kwargs=dict(axis=single_axis_arg),
                       vary_each_element=True, index_to_bnds={0: (-2, 2)},
                       atol=1e-5)
def test_cumsum_bkwd(): pass

