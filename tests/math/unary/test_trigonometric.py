from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory
from mygrad import sin, cos, tan, csc, sec, cot, sinc

import numpy as np


@fwdprop_test_factory(num_arrays=1, mygrad_func=sin, true_func=np.sin)
def test_sin_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=sin, true_func=np.sin, as_decimal=False)
def test_sin_backward(): pass


@fwdprop_test_factory(num_arrays=1, mygrad_func=sinc, true_func=np.sinc)
def test_sinc_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=sinc, true_func=np.sinc, as_decimal=False,
                       atol=1e-5)
def test_sinc_backward(): pass


@fwdprop_test_factory(num_arrays=1, mygrad_func=cos, true_func=np.cos)
def test_cos_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=cos, true_func=np.cos, as_decimal=False)
def test_cos_backward(): pass


@fwdprop_test_factory(num_arrays=1, mygrad_func=tan, true_func=np.tan,
                      index_to_bnds={0: (-np.pi/2 + 1e-5, np.pi/2 - 1e-5)})
def test_tan_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=tan, true_func=np.tan, as_decimal=False,
                       index_to_bnds={0: (-np.pi / 2 + 1e-5, np.pi / 2 - 1e-5)})
def test_tan_backward(): pass


@fwdprop_test_factory(num_arrays=1, mygrad_func=csc, true_func=lambda x: 1 / np.sin(x),
                      index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)})
def test_csc_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=csc, true_func=lambda x: 1 / np.sin(x),
                       as_decimal=False,
                       index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)},
                       atol=1e-4, rtol=1e-4)
def test_csc_backward(): pass


@fwdprop_test_factory(num_arrays=1, mygrad_func=sec, true_func=lambda x: 1 / np.cos(x),
                      index_to_bnds={0: (-np.pi/2 + 1e-5, np.pi/2 - 1e-5)})
def test_sec_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=sec, true_func=lambda x: 1 / np.cos(x),
                       index_to_bnds={0: (-np.pi/2 + 1e-5, np.pi/2 - 1e-5)}, atol=1e-4,
                       as_decimal=False)
def test_sec_backward(): pass


@fwdprop_test_factory(num_arrays=1, mygrad_func=cot, true_func=lambda x: 1 / np.tan(x),
                      index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)})
def test_cot_fwd(): pass


@backprop_test_factory(num_arrays=1, mygrad_func=cot, true_func=lambda x: 1 / np.tan(x),
                       index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)},
                       atol=1e-4, rtol=1e-4, as_decimal=False)
def test_cot_backward(): pass
