import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from mygrad import Tensor
from mygrad.nnet.initializers import normal


@given(
    dtype=hnp.unsigned_integer_dtypes()
    | hnp.integer_dtypes()
    | hnp.complex_number_dtypes()
)
def test_normal_dtype_validation(dtype):
    with pytest.raises(ValueError):
        normal(1, dtype=dtype)


@given(std=st.floats(-1000, 0, exclude_max=True))
def test_normal_std_validation(std):
    with pytest.raises(ValueError):
        normal(1, std=std)


_array_shapes = ((1000000,), (1000, 100, 10), (10, 10, 10, 10, 10, 10))


@given(
    shape=st.sampled_from(_array_shapes),
    mean=st.floats(-100, 100),
    std=st.floats(0, 5),
)
def test_normal_statistics(shape, mean, std):
    tensor = normal(shape, mean=mean, std=std)
    assert isinstance(tensor, Tensor)
    assert np.isclose(np.mean(tensor.data), mean, atol=1e-2)
    assert np.isclose(np.std(tensor.data), std, atol=1e-2)


@given(
    shape=hnp.array_shapes(min_dims=2),
    mean=st.floats(-100, 100),
    std=st.floats(0, 100),
    dtype=hnp.floating_dtypes(),
    constant=st.booleans(),
)
def test_normal(shape, mean, std, dtype, constant):
    tensor = normal(
        shape, mean=Tensor(mean), std=Tensor(std), dtype=dtype, constant=constant
    )
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.constant == constant
