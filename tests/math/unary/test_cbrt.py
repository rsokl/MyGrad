from tests.wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad.math import cbrt
import numpy as np


@fwdprop_test_factory(mygrad_func=cbrt, true_func=np.cbrt)
def test_cbrt_fwd(): pass


@backprop_test_factory(mygrad_func=cbrt, xbnds=(-100, 100), no_go=(0,))
def test_cbrt_backward(): pass
