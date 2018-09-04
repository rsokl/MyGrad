from mygrad import argmin, argmax
from mygrad.tensor_base import Tensor

import hypothesis.strategies as st
from numpy.testing import assert_equal
import numpy as np
from hypothesis import given

import numpy as np

@given(d=st.integers(min_value=3, max_value=10))
def test_argmin(d):
    dimensions = tuple(np.random.randint(1,6) for i in range(d)) 

    a = np.random.randint(-100,100, size=dimensions)
    axis = np.random.randint(0,d-1)
    assert_equal(Tensor(a).argmin(axis=axis), a.argmin(axis=axis))

@given(d=st.integers(min_value=3, max_value=10))
def test_argmax(d):
    dimensions = tuple(np.random.randint(1,6) for i in range(d)) 

    a = np.random.randint(-100,100, size=dimensions)
    axis = np.random.randint(0,d-1)
    assert_equal(Tensor(a).argmax(axis=axis), a.argmax(axis=axis))
