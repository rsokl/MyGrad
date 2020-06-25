from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
import pytest

from mygrad import Tensor
from mygrad.nnet.initializers import he_normal


@given(dtype=hnp.unsigned_integer_dtypes() | hnp.integer_dtypes() | hnp.complex_number_dtypes())
def test_glorot_normal_dtype_validation(dtype):
    with pytest.raises(ValueError):
        he_normal(1, 1, dtype=dtype)


@given(shape=hnp.array_shapes(max_dims=1))
def test_glorot_normal_input_validation(shape):
    with pytest.raises(ValueError):
        he_normal(shape)


_array_shapes = ((10000, 100), (1000, 100, 10), (10, 10, 10, 10, 10, 10))  # each 1 million elements
_valid_gains = (1, 5/3, np.sqrt(2), np.sqrt(2 / (1.01 ** 2)))  # most activations, tanh, relu, leaky


@given(shape=st.sampled_from(_array_shapes), gain=st.sampled_from(_valid_gains))
def test_glorot_normal_statistics(shape, gain):
    tensor = he_normal(shape, gain=gain)
    assert isinstance(tensor, Tensor)
    assert np.isclose(np.mean(tensor.data), 0, atol=1e-3)

    val = gain / np.sqrt(tensor.shape[1] * tensor[0, 0].size) / np.std(tensor.data)
    assert np.isclose(val, 1, atol=1e-3)


@given(
    shape=hnp.array_shapes(min_dims=2),
    gain=st.floats(0.1, 10),
    dtype=hnp.floating_dtypes(),
    constant=st.booleans(),
)
def test_glorot_normal(shape, gain, dtype, constant):
    tensor = he_normal(shape, gain=Tensor(gain), dtype=dtype, constant=constant)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.constant == constant
