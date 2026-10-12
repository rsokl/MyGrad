import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import settings
from numpy.testing import assert_allclose

import mygrad as mg
import mygrad.math.misc.funcs
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
    out = mygrad.math.misc.funcs.matmul(x1, x2, out=y[1:2])

    assert_allclose(y, [0.0, 2.0, 0.0])
    out.backward()
    assert_allclose(y.grad, [0.0, 1.0, 0.0])
    assert_allclose(x1.grad, [2.0])
    assert_allclose(x2.grad, [1.0])


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 3, 0), (0, 4)),
        ((2, 3, 5), (5, 0)),
        ((0, 3, 5), (5, 4)),
        ((2, 3, 5), (5, 4)),
    ],
)
def test_matmul_backprop_to_matrix_broadcast_over_batch(shape_a, shape_b):
    # exercises the fast path for the gradient of a 2D operand broadcast over
    # the leading dims of the other, including zero-size dims
    rng = np.random.default_rng(0)
    a, b = mg.tensor(rng.normal(size=shape_a)), mg.tensor(rng.normal(size=shape_b))
    grad = rng.normal(size=shape_a[:-1] + shape_b[-1:])
    mg.matmul(a, b).backward(grad)
    lead = list(range(len(shape_a) - 1))
    assert_allclose(b.grad, np.tensordot(a.data, grad, axes=(lead, lead)))
    assert_allclose(a.grad, grad @ b.data.T)
