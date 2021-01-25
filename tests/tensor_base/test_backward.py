import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg
from tests.custom_strategies import tensors


def test_simple_behavior():
    tensor = mg.Tensor([1.0, 2.0])

    # default behavior
    tensor.backward()
    assert_allclose(tensor.grad, [1.0, 1.0])

    # compatible with tensor, casts dtype, broadcasts
    tensor.backward(mg.Tensor(2))
    assert_allclose(tensor.grad, [2.0, 2.0])

    # works with array-like, broadcasts
    tensor.backward([3.0])
    assert_allclose(tensor.grad, [3.0, 3.0])

    with pytest.raises(ValueError):
        # incompatible shape
        tensor.backward(np.arange(3))


@given(
    tensor=tensors(
        dtype=hnp.floating_dtypes(endianness="="),
        fill=st.just(0),
        constant=False,
        shape=hnp.array_shapes(min_dims=0, min_side=0, max_dims=3),
    ),
    data=st.data(),
)
def test_tensor_backward_produces_grad_of_correct_dtype_and_shape(
    tensor: mg.Tensor, data: st.DataObject
):
    arrays_broadcastable_into_tensor = hnp.arrays(
        dtype=hnp.floating_dtypes(endianness="=") | hnp.integer_dtypes(endianness="="),
        shape=hnp.broadcastable_shapes(
            tensor.shape,
            min_side=min(tensor.shape, default=0),
            max_side=min(tensor.shape, default=0),
            max_dims=tensor.ndim,
        ),
    )

    grad = data.draw(st.none() | arrays_broadcastable_into_tensor, label="grad",)

    tensor.backward(grad)
    assert tensor.dtype == tensor.grad.dtype


@given(
    grad=hnp.array_shapes(min_dims=0, min_side=0, max_dims=4).map(
        lambda shape: np.full(shape, 1.0)
    ),
    tensor=hnp.array_shapes(min_dims=0, min_side=0, max_dims=4).map(
        lambda shape: mg.full(shape, 1.0)
    ),
)
def test_incompatible_grad_shape_raises(grad: np.ndarray, tensor: mg.Tensor):
    raises = False
    try:
        out = np.broadcast(tensor, grad)
        if out.shape != tensor.shape:
            raises = True
    except ValueError:
        raises = True

    if not raises:
        tensor.backward(grad)
        assert tensor.shape == tensor.grad.shape
        assert tensor.dtype == tensor.grad.dtype
    else:
        with pytest.raises(ValueError):
            tensor.backward(grad)
