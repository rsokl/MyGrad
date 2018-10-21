from tests.wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad import abs, absolute
import numpy as np


@fwdprop_test_factory(mygrad_func=abs, true_func=np.abs)
def test_abs_fwd(): pass


@backprop_test_factory(mygrad_func=abs, xbnds=(-100, 100), no_go=(0,))
def test_abs_backward(): pass


@fwdprop_test_factory(mygrad_func=absolute, true_func=np.absolute)
def test_absolute_fwd(): pass


@backprop_test_factory(mygrad_func=absolute, xbnds=(-100, 100), no_go=(0,))
def test_absolute_backward(): pass
