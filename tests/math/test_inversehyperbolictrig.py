from ..wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad.math import arcsinh, arccosh, arctanh, arccsch, arccoth
import numpy as np


@fwdprop_test_factory(mygrad_func=arcsinh, true_func=np.arcsinh)
def test_arcsinh_fwd(): pass


@backprop_test_factory(mygrad_func=arcsinh)
def test_arcsinh_backward(): pass


@fwdprop_test_factory(mygrad_func=arccosh, true_func=np.arccosh, xbnds=(1.001, 10))
def test_arccosh_fwd(): pass


@backprop_test_factory(mygrad_func=arccosh,  xbnds=(1.001, 10))
def test_arccosh_backward(): pass


@fwdprop_test_factory(mygrad_func=arctanh, true_func=np.arctanh, xbnds=(-.5, .5))
def test_arctanh_fwd(): pass


@backprop_test_factory(mygrad_func=arctanh,  xbnds=(-0.5, 0.5), no_go=(1,))
def test_arctanh_backward(): pass


@fwdprop_test_factory(mygrad_func=arccsch, true_func=lambda x: np.arcsinh(1 / x), xbnds=(1, 10))
def test_arccsch_fwd(): pass


@backprop_test_factory(mygrad_func=arccsch, xbnds=(1, 10))
def test_arccsch_backward(): pass


@fwdprop_test_factory(mygrad_func=arccoth, true_func=lambda x: np.arctanh(1 / x), xbnds=(5, 10))
def test_arccoth_fwd(): pass


@backprop_test_factory(mygrad_func=arccoth, xbnds=(5, 10))
def test_arccoth_backward(): pass