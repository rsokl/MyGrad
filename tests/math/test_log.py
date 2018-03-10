from mygrad.math import log, log2, log10
from ..wrappers.unary_func import backprop_test, fwdprop_test

import numpy as np


@fwdprop_test(mygrad_op=log, true_func=np.log, xbnds=[0, 100], no_go=(0,))
def test_log_fwd(): pass


@backprop_test(mygrad_op=log, xbnds=[0, 100], no_go=(0,))
def test_log_backward(): pass


@fwdprop_test(mygrad_op=log2, true_func=np.log2, xbnds=[0, 100], no_go=(0,))
def test_log2_fwd(): pass


@backprop_test(mygrad_op=log2, xbnds=[0, 100], no_go=(0,))
def test_log2_backward(): pass


@fwdprop_test(mygrad_op=log10, true_func=np.log10, xbnds=[0, 100], no_go=(0,))
def test_log10_fwd(): pass


@backprop_test(mygrad_op=log10, xbnds=[0, 100], no_go=(0,))
def test_log10_backward(): pass
