import numpy as np

from mygrad import sqrt
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=sqrt, true_func=np.sqrt, num_arrays=1, index_to_bnds={0: (0, 100)}
)
def test_sqrt_fwd():
    pass


@backprop_test_factory(
    mygrad_func=sqrt,
    true_func=np.sqrt,
    num_arrays=1,
    index_to_bnds={0: (1e-5, 100)},
    atol=1e-3,
)
def test_sqrt_backward():
    pass
