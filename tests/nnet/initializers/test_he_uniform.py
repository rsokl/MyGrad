import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from mygrad import Tensor
from mygrad.nnet.initializers import he_uniform


@given(shape=hnp.array_shapes(max_dims=1))
def test_he_uniform_input_validation(shape):
    with pytest.raises(ValueError):
        he_uniform(shape)


_array_shapes = ((10000, 100), (1000, 100, 10), (10, 10, 10, 10, 10, 10))
_valid_gains = (1, 5 / 3, np.sqrt(2), np.sqrt(2 / (1.01 ** 2)))


@given(shape=st.sampled_from(_array_shapes), gain=st.sampled_from(_valid_gains))
def test_he_uniform_statistics(shape, gain):
    tensor = he_uniform(shape, gain=gain)
    assert isinstance(tensor, Tensor)
    assert np.isclose(np.mean(tensor.data), 0, atol=1e-2)

    # check the bounds of the distribution hold
    factor = np.sqrt(3 / (tensor.shape[1] * (tensor[0, 0].size)))
    assert tensor.min() >= -gain / factor
    assert tensor.max() <= gain / factor

    # check that the distribution is roughly normal
    hist, _ = np.histogram(tensor.data, bins=100)
    assert (max(hist) - min(hist)) / np.mean(hist) < 0.1


@given(
    shape=hnp.array_shapes(min_dims=2),
    gain=st.floats(0.1, 10),
    dtype=hnp.floating_dtypes(),
    constant=st.booleans(),
)
def test_he_uniform(shape, gain, dtype, constant):
    tensor = he_uniform(shape, gain=Tensor(gain), dtype=dtype, constant=constant)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.constant == constant
