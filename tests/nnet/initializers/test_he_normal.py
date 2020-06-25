from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np

from mygrad import Tensor
from mygrad.nnet.initializers import he_normal


_array_shapes = ((10000, 100), (1000, 100, 10), (10, 10, 10, 10, 10, 10))  # each 1 million elements
_valid_gains = (1, 5/3, np.sqrt(2), np.sqrt(2 / (1.01 ** 2)))  # most activations, tanh, relu, leaky


@given(shape=st.sampled_from(_array_shapes), gain=st.sampled_from(_valid_gains))
def test_glorot_normal_statistics(shape, gain):
    tensor = he_normal(shape, gain=gain)
    assert isinstance(tensor, Tensor)
    assert np.isclose(np.mean(tensor.data), 0, atol=1e-3)

    val = gain / np.sqrt(tensor.shape[1] * tensor[0, 0].size) / np.std(tensor.data)
    assert np.isclose(val, 1, atol=1e-3)
