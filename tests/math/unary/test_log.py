from mygrad.math import log, log2, log10
from tests.wrappers.unary_func import backprop_test_factory, fwdprop_test_factory

import numpy as np


@fwdprop_test_factory(mygrad_func=log, true_func=np.log, xbnds=[0, 100], no_go=(0,))
def test_log_fwd(): pass


@backprop_test_factory(mygrad_func=log, xbnds=[0, 100], no_go=(0,))
def test_log_backward(): pass


@fwdprop_test_factory(mygrad_func=log2, true_func=np.log2, xbnds=[0, 100], no_go=(0,))
def test_log2_fwd(): pass


@backprop_test_factory(mygrad_func=log2, xbnds=[0, 100], no_go=(0,))
def test_log2_backward(): pass


@fwdprop_test_factory(mygrad_func=log10, true_func=np.log10, xbnds=[0, 100], no_go=(0,))
def test_log10_fwd(): pass


@backprop_test_factory(mygrad_func=log10, xbnds=[0, 100], no_go=(0,))
def test_log10_backward(): pass
