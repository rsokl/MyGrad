import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import settings
from numpy.testing import assert_allclose

import mygrad as mg
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


def test_matmul_with_target_tensor():
    y = mg.tensor([0.0, 0.0, 0.0])
    x1 = mg.tensor([1.0])
    x2 = mg.tensor([2.0])
    out = mg.matmul(x1, x2, out=y[1:2])

    assert_allclose(y, [0.0, 2.0, 0.0])
    out.backward()
    assert_allclose(y.grad, [0.0, 1.0, 0.0])
    assert_allclose(x1.grad, [2.0])
    assert_allclose(x2.grad, [1.0])
