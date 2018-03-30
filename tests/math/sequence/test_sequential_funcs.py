from ...wrappers.sequence_func import fwdprop_test_factory, backprop_test_factory
from mygrad import amax, amin, sum, mean
import mygrad

import numpy as np


@fwdprop_test_factory(mygrad_func=amax, true_func=np.amax)
def test_max_fwd(): pass


@backprop_test_factory(mygrad_func=amax, true_func=np.amax)
def test_max_bkwd(): pass


@fwdprop_test_factory(mygrad_func=amin, true_func=np.amin)
def test_min_fwd(): pass


@backprop_test_factory(mygrad_func=amin, true_func=np.amin)
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