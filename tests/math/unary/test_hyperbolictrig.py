from ..wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad.math import sinh, cosh, tanh, csch, sech, coth

import numpy as np


@fwdprop_test_factory(mygrad_func=sinh, true_func=np.sinh, xbnds=(-10, 10))
def test_sinh_fwd(): pass


@backprop_test_factory(mygrad_func=sinh, xbnds=(-10, 10))
def test_sinh_backward(): pass


@fwdprop_test_factory(mygrad_func=cosh, true_func=np.cosh, xbnds=(-10, 10))
def test_cosh_fwd(): pass


@backprop_test_factory(mygrad_func=cosh, xbnds=(-10, 10), atol=1e-5)
def test_cosh_backward(): pass


@fwdprop_test_factory(mygrad_func=tanh, true_func=np.tanh, xbnds=(-10, 10))
def test_tanh_fwd(): pass


@backprop_test_factory(mygrad_func=tanh, xbnds=(-10, 10), atol=1e-5)
def test_tanh_backward(): pass


@fwdprop_test_factory(mygrad_func=csch, true_func=lambda x: 1 / np.sinh(x), xbnds=(.001, 10))
def test_csch_fwd(): pass


@backprop_test_factory(mygrad_func=csch, xbnds=(.001, 10))
def test_csch_backward(): pass


@fwdprop_test_factory(mygrad_func=sech, true_func=lambda x: 1 / np.cosh(x), xbnds=(-10, 10))
def test_sech_fwd(): pass


@backprop_test_factory(mygrad_func=sech, xbnds=(.001, 10), atol=1e-5)
def test_sech_backward(): pass


@fwdprop_test_factory(mygrad_func=coth, true_func=lambda x: 1 / np.tanh(x), xbnds=(-10, 10), no_go=(0,))
def test_coth_fwd(): pass


@backprop_test_factory(mygrad_func=coth, xbnds=(.001, 10), atol=1e-5)
def test_coth_backward(): pass
