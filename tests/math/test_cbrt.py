from ..wrappers.unary_func import fwdprop_test, backprop_test
from mygrad.math import cbrt
import numpy as np


@fwdprop_test(mygrad_op=cbrt, true_func=np.cbrt)
def test_cbrt_fwd(): pass


@backprop_test(mygrad_op=cbrt, xbnds=(-100, 100), no_go=(0,))
def test_cbrt_backward(): pass
