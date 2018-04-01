from tests.wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad import sin, cos, tan, csc, sec, cot, sinc

import numpy as np


@fwdprop_test_factory(mygrad_func=sin, true_func=np.sin)
def test_sin_fwd(): pass


@backprop_test_factory(mygrad_func=sin)
def test_sin_backward(): pass


@fwdprop_test_factory(mygrad_func=sinc, true_func=np.sinc)
def test_sinc_fwd(): pass


@backprop_test_factory(mygrad_func=sinc)
def test_sinc_backward(): pass


@fwdprop_test_factory(mygrad_func=cos, true_func=np.cos)
def test_cos_fwd(): pass


@backprop_test_factory(mygrad_func=cos)
def test_cos_backward(): pass


@fwdprop_test_factory(mygrad_func=tan, true_func=np.tan,
                      xbnds=(-np.pi/2, np.pi/2),
                      no_go=(-np.pi/2, np.pi/2))
def test_tan_fwd(): pass


@backprop_test_factory(mygrad_func=tan, xbnds=(-np.pi / 1.5, np.pi / 1.5))
def test_tan_backward(): pass


@fwdprop_test_factory(mygrad_func=csc, true_func=lambda x: 1 / np.sin(x),
                      xbnds=(0, np.pi), no_go=(0, np.pi))
def test_csc_fwd(): pass


@backprop_test_factory(mygrad_func=csc, xbnds=(0.01, np.pi-.01), no_go=(0, np.pi), atol=1e-4, rtol=1e-4)
def test_csc_backward(): pass


@fwdprop_test_factory(mygrad_func=sec, true_func=lambda x: 1 / np.cos(x),
                      xbnds=(-np.pi/2, np.pi/2),
                      no_go=(-np.pi/2, np.pi/2))
def test_sec_fwd(): pass


@backprop_test_factory(mygrad_func=sec, xbnds=(-np.pi / 2.5, np.pi / 2.5), atol=1e-4)
def test_sec_backward(): pass


@fwdprop_test_factory(mygrad_func=cot, true_func=lambda x: 1 / np.tan(x),
                      xbnds=(0, np.pi),
                      no_go=(0, np.pi))
def test_cot_fwd(): pass


@backprop_test_factory(mygrad_func=cot, xbnds=(0, np.pi), no_go=(0, np.pi), atol=1e-4, rtol=1e-4)
def test_cot_backward(): pass
