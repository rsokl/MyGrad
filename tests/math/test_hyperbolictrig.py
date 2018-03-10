from ..wrappers.unary_func import fwdprop_test, backprop_test
from mygrad.math import sinh, cosh, tanh, csch, sech, coth

import numpy as np


@fwdprop_test(mygrad_op=sinh, true_func=np.sinh,  xbnds=(-10, 10))
def test_sinh_fwd(): pass


@backprop_test(mygrad_op=sinh, xbnds=(-10, 10))
def test_sinh_backward(): pass


@fwdprop_test(mygrad_op=cosh, true_func=np.cosh,  xbnds=(-10, 10))
def test_cosh_fwd(): pass


@backprop_test(mygrad_op=cosh, xbnds=(-10, 10), atol=1e-5)
def test_cosh_backward(): pass


@fwdprop_test(mygrad_op=tanh, true_func=np.tanh,  xbnds=(-10, 10))
def test_tanh_fwd(): pass


@backprop_test(mygrad_op=tanh, xbnds=(-10, 10), atol=1e-5)
def test_tanh_backward(): pass


@fwdprop_test(mygrad_op=csch, true_func=lambda x: 1 / np.sinh(x),  xbnds=(.001, 10))
def test_csch_fwd(): pass


@backprop_test(mygrad_op=csch, xbnds=(.001, 10))
def test_csch_backward(): pass


@fwdprop_test(mygrad_op=sech, true_func=lambda x: 1 / np.cosh(x),  xbnds=(-10, 10))
def test_sech_fwd(): pass


@backprop_test(mygrad_op=sech, xbnds=(.001, 10), atol=1e-5)
def test_sech_backward(): pass


@fwdprop_test(mygrad_op=coth, true_func=lambda x: 1 / np.tanh(x),  xbnds=(-10, 10), no_go=(0,))
def test_coth_fwd(): pass


@backprop_test(mygrad_op=coth, xbnds=(.001, 10), atol=1e-5)
def test_coth_backward(): pass
