import sys

import numpy as np

from mygrad.nnet.activations import selu
from mygrad.nnet.activations.selu import _ALPHA, _SCALE
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _np_selu(x):
    return _SCALE * np.where(x < 0, _ALPHA * (np.exp(x) - 1), x)


@fwdprop_test_factory(
    mygrad_func=selu,
    true_func=_np_selu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max), np.log(sys.float_info.max))},
)
def test_selu_fwd():
    pass


@backprop_test_factory(
    mygrad_func=selu,
    true_func=_np_selu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max), np.log(sys.float_info.max))},
)
def test_selu_bkwd():
    pass
