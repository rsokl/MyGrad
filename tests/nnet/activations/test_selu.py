import numpy as np

from mygrad import Tensor
from mygrad.nnet.activations import selu
from mygrad.nnet.activations.selu import _ALPHA, _SCALE
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _finite_params(arrs):
    if isinstance(arrs, Tensor):
        arrs = arrs.data
    return np.all(np.isfinite(_SCALE * _ALPHA * np.exp(arrs)))


def _np_selu(x):
    return _SCALE * np.where(x < 0, _ALPHA * (np.exp(x) - 1), x)


@fwdprop_test_factory(
    mygrad_func=selu,
    true_func=_np_selu,
    num_arrays=1,
    assumptions=_finite_params,
)
def test_selu_fwd():
    pass


@backprop_test_factory(
    mygrad_func=selu,
    true_func=_np_selu,
    num_arrays=1,
    assumptions=_finite_params,
)
def test_selu_bkwd():
    pass
