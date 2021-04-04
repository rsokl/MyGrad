from typing import Callable

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

from mygrad.random import (
    rand,
    randint,
    randn,
    random,
    random_sample,
    ranf,
    sample,
    seed,
)

shape_functions = [
    (np.random.sample, sample),
    (np.random.random_sample, random_sample),
    (np.random.ranf, ranf),
    (np.random.random, random),
]


@given(shape=hnp.array_shapes(max_side=4, max_dims=5), constant=st.booleans())
@pytest.mark.parametrize("np_function, mg_function", shape_functions)
def test_random_shape_funcs(
    np_function: Callable, mg_function: Callable, shape, constant: bool
):
    np.random.seed(0)
    arr = np_function(shape)
    seed(0)
    tens = mg_function(shape, constant=constant)
    assert_allclose(arr, tens.data)
    assert tens.constant is constant


unpacked_shape_functions = [(np.random.rand, rand), (np.random.randn, randn)]


@given(shape=hnp.array_shapes(max_side=4, max_dims=5), constant=st.booleans())
@pytest.mark.parametrize("np_function,mg_function", unpacked_shape_functions)
def test_unpacked_shape_funcs(np_function, mg_function, shape, constant: bool):
    np.random.seed(0)
    arr = np_function(*shape)
    seed(0)
    tens = mg_function(*shape, constant=constant)
    assert_allclose(arr, tens.data)
    assert tens.constant is constant


bound_shape_functions = [
    (np.random.randint, randint),
]


@given(
    shape=hnp.array_shapes(max_side=4, max_dims=5),
    m=st.integers(-10000, 10000),
    n=st.integers(-10000, 10000),
)
@pytest.mark.parametrize("np_function, mg_function", bound_shape_functions)
def test_bound_shape_functions(np_function, mg_function, m, n, shape):
    if m > n:
        m, n = n, m
    elif m == n:
        n = n + 1
    np.random.seed(0)
    arr = np_function(m, n, shape)
    seed(0)
    tens = mg_function(m, n, shape)
    assert_allclose(arr, tens.data)
