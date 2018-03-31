from ...wrappers.sequence_func import fwdprop_test_factory, backprop_test_factory

from pytest import raises

from mygrad import amax, amin, sum, mean, cumprod, cumsum
import mygrad

import numpy as np


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


@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-2, 2), max_dims=4, max_side=5)
def test_cumprod_bkwd(): pass


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


@backprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum,
                       no_keepdims=True, single_axis_only=True,
                       xbnds=(-2, 2), max_dims=4, max_side=5)
def test_cumsum_bkwd(): pass

