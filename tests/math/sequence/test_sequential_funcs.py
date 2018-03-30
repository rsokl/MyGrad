from ...wrappers.sequence_func import fwdprop_test_factory, backprop_test_factory
from mygrad import max, min, sum, mean

import numpy as np


@fwdprop_test_factory(mygrad_func=max, true_func=np.max)
def test_max_fwd(): pass


@backprop_test_factory(mygrad_func=max, true_func=np.max)
def test_max_bkwd(): pass


@fwdprop_test_factory(mygrad_func=min, true_func=np.min)
def test_min_fwd(): pass


@backprop_test_factory(mygrad_func=min, true_func=np.min)
def test_min_bkwd(): pass


@fwdprop_test_factory(mygrad_func=sum, true_func=np.sum)
def test_sum_fwd(): pass


@backprop_test_factory(mygrad_func=sum, true_func=np.sum)
def test_sum_bkwd(): pass


@fwdprop_test_factory(mygrad_func=mean, true_func=np.mean)
def test_mean_fwd(): pass


@backprop_test_factory(mygrad_func=mean, true_func=np.mean)
def test_mean_bkwd(): pass