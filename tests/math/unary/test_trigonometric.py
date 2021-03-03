import numpy as np

from mygrad import cot, csc, sec, sinc
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(num_arrays=1, mygrad_func=sinc, true_func=np.sinc)
def test_sinc_fwd():
    pass


@backprop_test_factory(num_arrays=1, mygrad_func=sinc, true_func=np.sinc, atol=1e-5)
def test_sinc_backward():
    pass


@fwdprop_test_factory(
    num_arrays=1,
    mygrad_func=csc,
    true_func=lambda x: 1 / np.sin(x),
    index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)},
)
def test_csc_fwd():
    pass


@backprop_test_factory(
    num_arrays=1,
    mygrad_func=csc,
    true_func=lambda x: 1 / np.sin(x),
    index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)},
)
def test_csc_backward():
    pass


@fwdprop_test_factory(
    num_arrays=1,
    mygrad_func=sec,
    true_func=lambda x: 1 / np.cos(x),
    index_to_bnds={0: (-np.pi / 2 + 1e-5, np.pi / 2 - 1e-5)},
)
def test_sec_fwd():
    pass


@backprop_test_factory(
    num_arrays=1,
    mygrad_func=sec,
    true_func=lambda x: 1 / np.cos(x),
    index_to_bnds={0: (-np.pi / 2 + 1e-5, np.pi / 2 - 1e-5)},
    atol=1e-4,
)
def test_sec_backward():
    pass


@fwdprop_test_factory(
    num_arrays=1,
    mygrad_func=cot,
    true_func=lambda x: 1 / np.tan(x),
    index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)},
)
def test_cot_fwd():
    pass


@backprop_test_factory(
    num_arrays=1,
    mygrad_func=cot,
    true_func=lambda x: 1 / np.tan(x),
    index_to_bnds={0: (0 + 1e-5, np.pi - 1e-5)},
)
def test_cot_backward():
    pass
