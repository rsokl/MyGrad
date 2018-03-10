from ..wrappers.unary_func import fwdprop_test, backprop_test
from mygrad.math import abs
import numpy as np


@fwdprop_test(mygrad_op=abs, true_func=np.abs)
def test_abs_fwd(): pass


@backprop_test(mygrad_op=abs, xbnds=(-100, 100), no_go=(0,))
def test_abs_backward(): pass
