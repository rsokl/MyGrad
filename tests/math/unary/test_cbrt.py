from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory
from mygrad import cbrt
import numpy as np


def _is_non_zero(x):
    return np.all(np.abs(x.data) > 1e-5)


@fwdprop_test_factory(mygrad_func=cbrt, true_func=np.cbrt, num_arrays=1)
def test_cbrt_fwd():
    pass


@backprop_test_factory(mygrad_func=cbrt, true_func=np.cbrt,
                       index_to_bnds={0: (-100, 100)}, num_arrays=1,
                       assumptions=_is_non_zero,
                       as_decimal=False, atol=1e-5)
def test_cbrt_backward():
    pass
