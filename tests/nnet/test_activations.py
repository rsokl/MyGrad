import numpy as np

from mygrad.nnet.activations import relu
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=relu, true_func=lambda x: np.maximum(x, 0), num_arrays=1
)
def test_relu_fwd():
    pass


def _away_from_zero(*arrs, **kwargs):
    x = arrs[0]
    return np.all(np.abs(x.data) > 1e-8)


@backprop_test_factory(
    mygrad_func=relu,
    true_func=lambda x: np.maximum(x, 0),
    num_arrays=1,
    assumptions=_away_from_zero,
)
def test_relu_bkwd():
    pass
