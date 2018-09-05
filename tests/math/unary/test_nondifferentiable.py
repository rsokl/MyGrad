from mygrad as mg
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
    tensor = Tensor(a)
    # tensor input
    assert_allclose(mg.argmin(tensor), np.argmin(a, axis=axis))
    
    # tensor method
    assert_allclose(tensor.argmin(axis=axis), a.argmin(axis=axis))
    
    # array input
    assert_allclose(mg.argmin(a, axis=axis), np.argmin(a, axis=axis))

@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=dtype_strat_numpy),
       data=st.data())
def test_argmax(a, data):
    axis = data.draw(valid_axes(ndim=a.ndim, single_axis_only=True), label="axis")
    tensor = Tensor(a)
    
    # tensor input
    assert_allclose(mg.argmax(tensor), np.argmax(a, axis=axis))
    
    # tensor method
    assert_allclose(tensor.argmax(axis=axis), a.argmax(axis=axis))
    
    # array input
    assert_allclose(mg.argmax(a, axis=axis), np.argmax(a, axis=axis))
