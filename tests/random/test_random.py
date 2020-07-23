import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose
import pytest

from mygrad.random import *
from mygrad.tensor_base import Tensor

dtype_strat_numpy = st.sampled_from(
    (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
)

shape_functions = [
    (np.random.sample, sample),
    (np.random.random_sample, random_sample),
    (np.random.ranf, ranf),
    (np.random.random, random),
]


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
@pytest.mark.parametrize("np_function,mg_function", shape_functions)
def test_random_shape_funcs(np_function, mg_function, a):
    shape = a.shape
    np.random.seed(0)
    arr = np_function(shape)
    np.random.seed(0)
    tens = mg_function(shape)
    assert_allclose(arr, tens.data)


unpacked_shape_functions = [(np.random.rand, rand), (np.random.randn, randn)]


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
@pytest.mark.parametrize("np_function,mg_function", unpacked_shape_functions)
def test_random_shape_funcs(np_function, mg_function, a):
    shape = a.shape
    np.random.seed(0)
    arr = np_function(*shape)
    np.random.seed(0)
    tens = mg_function(*shape)
    assert_allclose(arr, tens.data)


bound_shape_functions = [
    (np.random.randint, randint),
    (np.random.random_integers, random_integers),
]


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    ),
    m=st.integers(-10000, 10000),
    n=st.integers(-10000, 10000),
)
@pytest.mark.parametrize("np_function,mg_function", bound_shape_functions)
def test_bound_shape_functions(np_function, mg_function, m, n, a):
    shape = a.shape
    shape = a.shape
    if m > n:
        m, n = n, m
    elif m == n:
        n = n + 1
    np.random.seed(0)
    arr = np_function(m, n, shape)
    np.random.seed(0)
    tens = mg_function(m, n, shape)
    assert_allclose(arr, tens.data)
