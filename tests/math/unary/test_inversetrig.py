import numpy as np

from mygrad import arccot, arccsc, arcsec
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=arccsc,
    true_func=lambda x: np.arcsin(1 / x),
    index_to_bnds={0: (-10.1, -1.1)},
    num_arrays=1,
)
def test_arccsc_fwd():
    pass


@backprop_test_factory(
    mygrad_func=arccsc,
    true_func=lambda x: np.arcsin(1 / x),
    index_to_bnds={0: (1.1, 100.0)},
    num_arrays=1,
)
def test_arccsc_backward(data):
    pass


@fwdprop_test_factory(
    mygrad_func=arcsec,
    true_func=lambda x: np.arccos(1 / x),
    index_to_bnds={0: (-10.1, -1.1)},
    num_arrays=1,
)
def test_arcsec_fwd():
    pass


@backprop_test_factory(
    mygrad_func=arcsec,
    true_func=lambda x: np.arccos(1 / x),
    index_to_bnds={0: (1.1, 100.0)},
    num_arrays=1,
)
def test_arcsec_backward(data):
    pass


@fwdprop_test_factory(
    mygrad_func=arccot,
    true_func=lambda x: np.arctan(1 / x),
    index_to_bnds={0: (0.1, 10)},
    num_arrays=1,
)
def test_arccot_fwd():
    pass


@backprop_test_factory(
    mygrad_func=arccot,
    true_func=lambda x: np.arctan(1 / x),
    index_to_bnds={0: (0.1, 10)},
    num_arrays=1,
)
def test_arccot_backward(data):
    pass
