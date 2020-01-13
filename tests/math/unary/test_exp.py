import sys

import numpy as np

from mygrad import exp, exp2, expm1
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=exp,
    true_func=np.exp,
    num_arrays=1,
    index_to_bnds={0: (None, np.log(sys.float_info.max))},
)
def test_exp_fwd():
    pass


@backprop_test_factory(
    mygrad_func=exp,
    true_func=np.exp,
    num_arrays=1,
    index_to_bnds={0: (None, np.log(sys.float_info.max) / 10)},
)
def test_exp_backward():
    pass


@fwdprop_test_factory(
    mygrad_func=expm1,
    true_func=np.expm1,
    num_arrays=1,
    index_to_bnds={0: (None, np.log(sys.float_info.max))},
)
def test_expm1_fwd():
    pass


@backprop_test_factory(
    mygrad_func=expm1,
    true_func=lambda x: np.exp(x) - 1,
    num_arrays=1,
    index_to_bnds={0: (None, np.log(sys.float_info.max) / 10)},
)
def test_expm1_backward():
    pass


@fwdprop_test_factory(
    mygrad_func=exp2,
    true_func=np.exp2,
    num_arrays=1,
    index_to_bnds={0: (None, np.log2(sys.float_info.max))},
)
def test_exp2_fwd():
    pass


@backprop_test_factory(
    mygrad_func=exp2,
    true_func=np.exp2,
    num_arrays=1,
    index_to_bnds={0: (None, np.log2(sys.float_info.max) / 10)},
)
def test_exp2_backward():
    pass
