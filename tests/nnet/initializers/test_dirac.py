from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np

from mygrad import Tensor
from mygrad.nnet.initializers import dirac


@given(shape=hnp.array_shapes(min_dims=2))
def test_dirac_dimensions(shape):
    tensor = dirac(shape)
    # each dimension should have at most a single 1
    assert isinstance(tensor, Tensor)
    assert np.all(tensor.sum(axis=tuple(i for i in range(1, len(shape)))) <= 1)
