import numpy as np

from mygrad import cbrt
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _is_non_zero(x):
    return np.all(np.abs(x.data) > 1e-5)


@fwdprop_test_factory(mygrad_func=cbrt, true_func=np.cbrt, num_arrays=1)
def test_cbrt_fwd():
    pass


@backprop_test_factory(
    mygrad_func=cbrt,
    true_func=np.cbrt,
    index_to_bnds={0: (-100, 100)},
    num_arrays=1,
    assumptions=_is_non_zero,
    atol=1e-5,
    rtol=1e-5,
    use_finite_difference=True,
    h=1e-8,
)
def test_cbrt_backward():
    pass
