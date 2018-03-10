from ..wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad.math import sqrt
import numpy as np


@fwdprop_test_factory(mygrad_op=sqrt, true_func=np.sqrt, xbnds=(0, 100))
def test_sqrt_fwd(): pass


@backprop_test_factory(mygrad_op=sqrt, true_func=np.sqrt, xbnds=(0, 100), no_go=(0,), atol=1e-3)
def test_sqrt_backward(): pass
