import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import settings

from mygrad import matmul
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=matmul,
    true_func=np.matmul,
    shapes=hnp.mutually_broadcastable_shapes(
        signature="(n?,k),(k,m?)->(n?,m?)", max_side=4
    ),
)
def test_matmul_fwd():
    pass


@settings(max_examples=500)
@backprop_test_factory(
    mygrad_func=matmul,
    true_func=np.matmul,
    shapes=hnp.mutually_broadcastable_shapes(
        signature="(n?,k),(k,m?)->(n?,m?)", max_side=4
    ),
    vary_each_element=True,
)
def test_matmul_bkwd():
    pass
