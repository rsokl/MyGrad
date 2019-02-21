from mygrad import log, log2, log10, log1p
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory

import numpy as np


@fwdprop_test_factory(mygrad_func=log, true_func=np.log, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log_fwd():
    pass


@backprop_test_factory(mygrad_func=log, true_func=np.log, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log_backward():
    pass


@fwdprop_test_factory(mygrad_func=log2, true_func=np.log2, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log2_fwd():
    pass


@backprop_test_factory(mygrad_func=log2, true_func=np.log2, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log2_backward():
    pass


@fwdprop_test_factory(mygrad_func=log10, true_func=np.log10, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log10_fwd():
    pass


@backprop_test_factory(mygrad_func=log10, true_func=np.log10, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log10_backward():
    pass


@fwdprop_test_factory(mygrad_func=log1p, true_func=np.log1p, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log1p_fwd():
    pass


@backprop_test_factory(mygrad_func=log1p, true_func=np.log1p, index_to_bnds={0: (1e-5, 100)}, num_arrays=1)
def test_log1p_backward():
    pass
