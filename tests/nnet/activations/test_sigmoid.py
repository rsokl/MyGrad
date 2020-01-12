import numpy as np

from mygrad.nnet.activations import sigmoid
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=sigmoid, true_func=lambda x: 1 / (1 + np.exp(-x)), num_arrays=1
)
def test_sigmoid_fwd():
    pass


def _not_large(*arrs, **kwargs):
    x = arrs[0]
    return np.all(np.abs(x.data) < 500)


@backprop_test_factory(
    mygrad_func=sigmoid, true_func=lambda x: 1 / (1 + np.exp(-x)), num_arrays=1, index_to_bnds={0: (-500, 500)},
)
def test_sigmoid_bkwd():
    pass
