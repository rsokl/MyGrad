import numpy as np

from mygrad import negative, positive, reciprocal, square
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _is_non_zero(x):
    return np.all(np.abs(x.data) > 1e-3)


@fwdprop_test_factory(mygrad_func=positive, true_func=np.positive, num_arrays=1)
def test_positive_fwd():
    pass


@backprop_test_factory(mygrad_func=positive, true_func=np.positive, num_arrays=1)
def test_positive_backward():
    pass


@fwdprop_test_factory(mygrad_func=negative, true_func=np.negative, num_arrays=1)
def test_negative_fwd():
    pass


@backprop_test_factory(mygrad_func=negative, true_func=np.negative, num_arrays=1)
def test_negative_backward():
    pass


@fwdprop_test_factory(
    mygrad_func=reciprocal,
    true_func=np.reciprocal,
    num_arrays=1,
    assumptions=_is_non_zero,
)
def test_reciprocal_fwd():
    pass


@backprop_test_factory(
    mygrad_func=reciprocal,
    true_func=np.reciprocal,
    num_arrays=1,
    assumptions=_is_non_zero,
    atol=1e-5,
    rtol=1e-5,
)
def test_reciprocal_backward():
    pass


@fwdprop_test_factory(mygrad_func=square, true_func=np.square, num_arrays=1)
def test_square_fwd():
    pass


@backprop_test_factory(mygrad_func=square, true_func=np.square, num_arrays=1)
def test_square_backward():
    pass
