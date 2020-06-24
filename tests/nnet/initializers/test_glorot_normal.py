from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
import pytest

from mygrad import Tensor
from mygrad.nnet.initializers import glorot_normal


@given(dtype=hnp.unsigned_integer_dtypes() | hnp.integer_dtypes() | hnp.complex_number_dtypes())
def test_glorot_normal_dtype_validation(dtype):
    with pytest.raises(ValueError):
        glorot_normal(1, 1, dtype=dtype)


@given(shape=hnp.array_shapes(max_dims=1))
def test_glorot_normal_input_validation(shape):
    with pytest.raises(ValueError):
        glorot_normal(shape)


_array_shapes = ((100_00, 100), (1000, 100, 10), (10, 10, 10, 10, 10, 10))  # each 1 million elements


@given(shape=st.sampled_from(_array_shapes), gain=st.floats(0.1, 10))
def test_glorot_normal_statistics(shape, gain):
    tensor = glorot_normal(shape)
    assert isinstance(tensor, Tensor)
    assert np.isclose(np.mean(tensor.data), 0, atol=1e-3)

    fan_in = tensor.shape[1] * (tensor.shape[-1] if tensor.ndim > 2 else 1)
    fan_out = tensor.shape[0] * (tensor.shape[-1] if tensor.ndim > 2 else 1)
    val = np.sqrt(2 / (fan_in + fan_out)) / np.std(tensor.data)
    assert np.isclose(val, 1, atol=1e-3)


@given(
    shape=hnp.array_shapes(min_dims=2),
    gain=st.floats(allow_infinity=False, allow_nan=False),
    dtype=hnp.floating_dtypes(),
    constant=st.booleans(),
)
def test_glorot_normal(shape, gain, dtype, constant):
    tensor = glorot_normal(shape, gain=Tensor(gain), dtype=dtype, constant=constant)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.constant == constant
