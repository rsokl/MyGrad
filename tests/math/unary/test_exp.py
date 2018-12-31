from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory
from mygrad import exp, expm1, exp2
import numpy as np


@fwdprop_test_factory(mygrad_func=exp, true_func=np.exp, num_arrays=1)
def test_exp_fwd(): pass


@backprop_test_factory(mygrad_func=exp, true_func=np.exp, num_arrays=1)
def test_exp_backward(): pass


@fwdprop_test_factory(mygrad_func=expm1, true_func=np.expm1, num_arrays=1)
def test_expm1_fwd(): pass


@backprop_test_factory(mygrad_func=expm1, true_func=lambda x: np.exp(x) - 1, num_arrays=1)
def test_expm1_backward(): pass


@fwdprop_test_factory(mygrad_func=exp2, true_func=np.exp2, num_arrays=1)
def test_exp2_fwd(): pass


@backprop_test_factory(mygrad_func=exp2, true_func=np.exp2, num_arrays=1, as_decimal=False)
def test_exp2_backward(): pass
