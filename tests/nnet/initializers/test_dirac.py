import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from mygrad import Tensor
from mygrad.nnet.initializers import dirac


@given(shape=hnp.array_shapes(max_dims=1))
def test_dirac_bad_dimensions(shape):
    with pytest.raises(ValueError):
        dirac(shape)


@given(shape=hnp.array_shapes(min_dims=2))
def test_dirac_dimensions(shape):
    tensor = dirac(shape)
    # each dimension should have at most a single 1
    assert isinstance(tensor, Tensor)
    assert np.all(tensor.sum(axis=tuple(i for i in range(1, len(shape)))) <= 1)


@given(
    matrix=hnp.arrays(
        shape=hnp.array_shapes(min_dims=2, max_dims=2),
        elements=st.floats(allow_infinity=False, allow_nan=False),
        dtype=float,
    ),
)
def test_dirac_matmul(matrix):
    weight = dirac(matrix.shape[0], matrix.shape[0])
    output = weight @ matrix
    assert isinstance(output, Tensor)
    assert np.allclose(output.data, matrix, equal_nan=True)
