from ..wrappers.unary_func import fwdprop_test, backprop_test
from mygrad.math import sqrt
import numpy as np


@fwdprop_test(mygrad_op=sqrt, true_func=np.sqrt, xbnds=(0, 100))
def test_sqrt_fwd(): pass


@backprop_test(mygrad_op=sqrt, true_func=np.sqrt, xbnds=(0, 100), no_go=(0,), atol=1e-3)
def test_sqrt_backward(): pass
