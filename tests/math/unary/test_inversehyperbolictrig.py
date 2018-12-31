from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory
from mygrad import arcsinh, arccosh, arctanh, arccsch, arccoth
import numpy as np


def _is_non_zero(x):
    return np.all(np.abs(x.data) > 1e-8)


@fwdprop_test_factory(mygrad_func=arcsinh, num_arrays=1,  true_func=np.arcsinh)
def test_arcsinh_fwd():
    pass


@backprop_test_factory(mygrad_func=arcsinh, true_func=np.arcsinh, num_arrays=1, as_decimal=False)
def test_arcsinh_backward():
    pass


@fwdprop_test_factory(mygrad_func=arccosh, num_arrays=1,  true_func=np.arccosh,
                      index_to_bnds={0: (1.001, 10)})
def test_arccosh_fwd():
    pass


@backprop_test_factory(mygrad_func=arccosh, true_func=np.arccosh, num_arrays=1,
                       index_to_bnds={0: (1.001, 10)}, as_decimal=False)
def test_arccosh_backward():
    pass


@fwdprop_test_factory(mygrad_func=arctanh, true_func=np.arctanh,
                      num_arrays=1,  index_to_bnds={0: (-.5, .5)})
def test_arctanh_fwd():
    pass


@backprop_test_factory(mygrad_func=arctanh, true_func=np.arctanh, num_arrays=1,
                       index_to_bnds={0: (-0.5, 0.5)},
                       assumptions=_is_non_zero, as_decimal=False)
def test_arctanh_backward():
    pass


@fwdprop_test_factory(mygrad_func=arccsch, num_arrays=1,
                      true_func=lambda x: np.arcsinh(1 / x), index_to_bnds={0: (1, 10)})
def test_arccsch_fwd():
    pass


@backprop_test_factory(mygrad_func=arccsch, true_func=lambda x: np.arcsinh(1 / x),
                       num_arrays=1,  index_to_bnds={0: (1, 10)}, as_decimal=False)
def test_arccsch_backward():
    pass


@fwdprop_test_factory(mygrad_func=arccoth, num_arrays=1,
                      true_func=lambda x: np.arctanh(1 / x), index_to_bnds={0: (5, 10)})
def test_arccoth_fwd():
    pass


@backprop_test_factory(mygrad_func=arccoth, true_func=lambda x: np.arctanh(1 / x),
                       num_arrays=1,  index_to_bnds={0: (5, 10)}, as_decimal=False)
def test_arccoth_backward():
    pass