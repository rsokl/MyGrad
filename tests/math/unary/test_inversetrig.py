from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory

from mygrad import arcsin, arccos, arctan, arccsc, arcsec, arccot
import numpy as np


@fwdprop_test_factory(mygrad_func=arcsin, true_func=np.arcsin,
                      index_to_bnds={0: (-.9, .9)}, num_arrays=1)
def test_arcsin_fwd():
    pass


@backprop_test_factory(mygrad_func=arcsin, true_func=np.arcsin,
                       index_to_bnds={0: (-.9, .9)}, num_arrays=1,
                       )
def test_arcsin_backward(data):
    pass


@fwdprop_test_factory(mygrad_func=arccos, true_func=np.arccos,
                      index_to_bnds={0: (-.9, .9)}, num_arrays=1)
def test_arccos_fwd():
    pass


@backprop_test_factory(mygrad_func=arccos, true_func=np.arccos,
                       index_to_bnds={0: (-.9, .9)}, num_arrays=1,
                       )
def test_arccos_backward(data):
    pass


@fwdprop_test_factory(mygrad_func=arctan, true_func=np.arctan,
                      index_to_bnds={0: (.1, 10.)}, num_arrays=1)
def test_arctan_fwd():
    pass


@backprop_test_factory(mygrad_func=arctan, true_func=np.arctan,
                       index_to_bnds={0: (.1, 10.)}, num_arrays=1,
                       )
def test_arctan_backward(data):
    pass


@fwdprop_test_factory(mygrad_func=arccsc, true_func=lambda x: np.arcsin(1 / x),
                      index_to_bnds={0: (-10.1, -1.1)}, num_arrays=1)
def test_arccsc_fwd():
    pass


@backprop_test_factory(mygrad_func=arccsc, true_func=lambda x: np.arcsin(1 / x),
                       index_to_bnds={0: (1.1, 100.)}, num_arrays=1,
                       )
def test_arccsc_backward(data):
    pass


@fwdprop_test_factory(mygrad_func=arcsec, true_func=lambda x: np.arccos(1 / x),
                      index_to_bnds={0: (-10.1, -1.1)}, num_arrays=1)
def test_arcsec_fwd():
    pass


@backprop_test_factory(mygrad_func=arcsec, true_func=lambda x: np.arccos(1 / x),
                       index_to_bnds={0: (1.1, 100.)}, num_arrays=1)
def test_arcsec_backward(data):
    pass


@fwdprop_test_factory(mygrad_func=arccot, true_func=lambda x: np.arctan(1 / x),
                      index_to_bnds={0: (0.1, 10)}, num_arrays=1)
def test_arccot_fwd():
    pass


@backprop_test_factory(mygrad_func=arccot, true_func=lambda x: np.arctan(1 / x),
                       index_to_bnds={0: (0.1, 10)}, num_arrays=1)
def test_arccot_backward(data):
    pass
