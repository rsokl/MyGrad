from mygrad import argmin, argmax
from mygrad.tensor_base import Tensor

import hypothesis.strategies as st
from numpy.testing import assert_allclose
import numpy as np

from ...custom_strategies import valid_axes
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np

dtype_strat_numpy = st.sampled_from((np.int8, np.int16, np.int32, np.int64,
                                     np.float16, np.float32, np.float64))

@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=dtype_strat_numpy),
       data=st.data())
def test_argmin(a, data):
    axis = data.draw(valid_axes(ndim=a.ndim, single_axis_only=True), label="axis")

    assert_allclose(Tensor(a).argmin(axis=axis), a.argmin(axis=axis))

@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=dtype_strat_numpy),
       data=st.data())
def test_argmax(a, data):
    axis = data.draw(valid_axes(ndim=a.ndim, single_axis_only=True), label="axis")
    
    assert_allclose(Tensor(a).argmax(axis=axis), a.argmax(axis=axis))