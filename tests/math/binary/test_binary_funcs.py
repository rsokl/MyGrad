""" Test all binary arithmetic operations, checks for appropriate broadcast behavior"""
from ...wrappers.uber import fwdprop_test_factory, backprop_test_factory

from mygrad import add, subtract, multiply, divide, power, logaddexp
from mygrad import logaddexp2

import numpy as np


@fwdprop_test_factory(mygrad_func=add, true_func=np.add, num_arrays=2)
def test_add_fwd(): pass


@backprop_test_factory(mygrad_func=add, true_func=np.add, num_arrays=2,
                       atol=1e-4, rtol=1e-4)
def test_add_bkwd(): pass


@fwdprop_test_factory(mygrad_func=subtract, true_func=np.subtract, num_arrays=2)
def test_subtract_fwd(): pass


@backprop_test_factory(mygrad_func=subtract, true_func=np.subtract, num_arrays=2,
                       atol=1e-4, rtol=1e-4)
def test_subtract_bkwd(): pass


@fwdprop_test_factory(mygrad_func=multiply, true_func=np.multiply, num_arrays=2)
def test_multiply_fwd(): pass


@backprop_test_factory(mygrad_func=multiply, true_func=np.multiply, atol=1e-4, rtol=1e-4, num_arrays=2)
def test_multiply_bkwd(): pass


@fwdprop_test_factory(mygrad_func=divide, true_func=np.divide, index_to_bnds={1: (1, 10)}, num_arrays=2)
def test_divide_fwd(): pass


@backprop_test_factory(mygrad_func=divide, true_func=np.divide, index_to_bnds={1: (1, 10)}, num_arrays=2,
                       atol=1e-4, rtol=1e-4)
def test_divide_bkwd(): pass


@fwdprop_test_factory(mygrad_func=power, true_func=np.power,
                      index_to_bnds={0: (1, 10), 1: (-3, 3)},
                      num_arrays=2)
def test_power_fwd(): pass


@backprop_test_factory(mygrad_func=power, true_func=np.power,
                       index_to_bnds={0: (1, 10), 1: (-3, 3)},
                       num_arrays=2, atol=1e-4, rtol=1e-4)
def test_power_bkwd(): pass


@fwdprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp, num_arrays=2)
def test_logaddexp_fwd(): pass


@backprop_test_factory(mygrad_func=logaddexp, true_func=np.logaddexp, num_arrays=2,
                       as_decimal=False, atol=1e-4, rtol=1e-4)
def test_logaddexp_bkwd(): pass


@fwdprop_test_factory(mygrad_func=logaddexp2, true_func=np.logaddexp2, num_arrays=2)
def test_logaddexp2_fwd(): pass


@backprop_test_factory(mygrad_func=logaddexp2, true_func=np.logaddexp2, num_arrays=2,
                       as_decimal=False, atol=1e-4, rtol=1e-4)
def test_logaddexp2_bkwd(): pass
