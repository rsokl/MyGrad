from ..wrappers.unary_func import fwdprop_test_factory, backprop_test_factory
from mygrad.math import sin, cos, tan, csc, sec, cot

import numpy as np


@fwdprop_test_factory(mygrad_op=sin, true_func=np.sin)
def test_sin_fwd(): pass


@backprop_test_factory(mygrad_op=sin)
def test_sin_backward(): pass


@fwdprop_test_factory(mygrad_op=cos, true_func=np.cos)
def test_cos_fwd(): pass


@backprop_test_factory(mygrad_op=cos)
def test_cos_backward(): pass


@fwdprop_test_factory(mygrad_op=tan, true_func=np.tan,
                      xbnds=(-np.pi/2, np.pi/2),
                      no_go=(-np.pi/2, np.pi/2))
def test_tan_fwd(): pass


@backprop_test_factory(mygrad_op=tan, xbnds=(-np.pi / 1.5, np.pi / 1.5))
def test_tan_backward(): pass


@fwdprop_test_factory(mygrad_op=csc, true_func=lambda x: 1 / np.sin(x),
                      xbnds=(0, np.pi), no_go=(0, np.pi))
def test_csc_fwd(): pass


@backprop_test_factory(mygrad_op=csc, xbnds=(0, np.pi), no_go=(0, np.pi))
def test_csc_backward(): pass


@fwdprop_test_factory(mygrad_op=sec, true_func=lambda x: 1 / np.cos(x),
                      xbnds=(-np.pi/2, np.pi/2),
                      no_go=(-np.pi/2, np.pi/2))
def test_sec_fwd(): pass


@backprop_test_factory(mygrad_op=sec, xbnds=(-np.pi / 2, np.pi / 2), no_go=(-np.pi / 2, np.pi / 2))
def test_sec_backward(): pass


@fwdprop_test_factory(mygrad_op=cot, true_func=lambda x: 1 / np.tan(x),
                      xbnds=(0, np.pi),
                      no_go=(0, np.pi))
def test_cot_fwd(): pass


@backprop_test_factory(mygrad_op=cot, xbnds=(0, np.pi), no_go=(0, np.pi))
def test_cot_backward(): pass
