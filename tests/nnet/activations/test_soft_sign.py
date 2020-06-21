import numpy as np

from mygrad.nnet.activations import soft_sign
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _np_soft_sign(x):
    return x / (1 + np.abs(x))


@fwdprop_test_factory(mygrad_func=soft_sign, true_func=_np_soft_sign, num_arrays=1)
def test_soft_sign_fwd():
    pass


def _away_from_zero(*arrs, **kwargs):
    x = arrs[0]
    return np.all(np.abs(x.data) > 1e-8)


@backprop_test_factory(
    mygrad_func=soft_sign,
    true_func=_np_soft_sign,
    num_arrays=1,
    assumptions=_away_from_zero,
    atol=1e-5,
    use_finite_difference=True,
    h=1e-8,
)
def test_soft_sign_bkwd():
    pass
