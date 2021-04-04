import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given
from numpy.testing import assert_array_equal

import mygrad as mg
from tests.custom_strategies import tensors


def test_backprop_through_non_scalar_matches_documented_behavior():
    tensor1 = mg.Tensor([2.0, 4.0, 8.0])
    (tensor1 * tensor1[::-1]).backward()  # behaves like ℒ = x0*x2 + x1*x1 + x2*x0

    tensor2 = mg.Tensor([2.0, 4.0, 8.0])
    mg.sum(tensor2 * tensor2[::-1]).backward()
    assert_array_equal(tensor1.grad, tensor2.grad)


@given(
    shapes=hnp.mutually_broadcastable_shapes(
        signature="(n?,k),(k,m?)->(n?,m?)", max_side=4
    ),
    data=st.data(),
)
def test_backprop_through_unsummed_matmul_matches_summed_matmul(
    shapes: hnp.BroadcastableShapes, data: st.DataObject
):
    shape_a, shape_b = shapes.input_shapes
    tensor_a1 = data.draw(
        tensors(shape=shape_a, elements=st.floats(-10, 10)), label="tensor-a"
    )
    tensor_a2 = tensor_a1.copy()

    tensor_b1 = data.draw(
        tensors(shape=shape_b, elements=st.floats(-10, 10)), label="tensor-b"
    )
    tensor_b2 = tensor_b1.copy()

    (tensor_a1 @ tensor_b1).backward()
    (tensor_a2 @ tensor_b2).sum().backward()
    assert_array_equal(tensor_a1.grad, tensor_a2.grad)
    assert_array_equal(tensor_b1.grad, tensor_b2.grad)
