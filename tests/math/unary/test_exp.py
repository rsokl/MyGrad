from ..wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad.math import exp
import numpy as np


@fwdprop_test_factory(mygrad_func=exp, true_func=np.exp)
def test_abs_fwd(): pass


@backprop_test_factory(mygrad_func=exp, xbnds=(-100, 100))
def test_abs_backward(): pass