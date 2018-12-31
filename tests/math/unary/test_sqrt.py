from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory
from mygrad import sqrt
import numpy as np


@fwdprop_test_factory(mygrad_func=sqrt, true_func=np.sqrt, num_arrays=1, index_to_bnds={0: (0, 100)})
def test_sqrt_fwd():
    pass


@backprop_test_factory(mygrad_func=sqrt, true_func=np.sqrt, num_arrays=1, index_to_bnds={0: (1e-5, 100)},
                       atol=1e-3)
def test_sqrt_backward():
    pass
