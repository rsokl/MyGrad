import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

from mygrad.random import *
from mygrad.tensor_base import Tensor

dtype_strat_numpy = st.sampled_from(
    (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
)


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
def test_rand(a):
    shape = a.shape
    b = rand(*shape)
    assert b.shape == a.shape
    assert (0 <= b.data).all() and (b.data<= 1).all()


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
def test_randn(a):
    shape = a.shape
    b = randn(*shape)
    assert b.shape == a.shape


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    ),
    m = st.integers(-10000, 10000), n=st.integers(-10000, 10000)
)
def test_randint(a, m, n):
    shape = a.shape
    if m > n:
        m, n = n, m
    elif m == n:
        n = n+1
    b = randint(m, n, shape)
    assert b.shape == a.shape
    assert (m <= b.data).all() and (b.data< n).all()


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
def test_random(a):
    shape = a.shape
    b = random(shape)
    assert b.shape == a.shape
    assert (0 <= b.data).all() and (b.data<= 1).all()


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    ),
    m = st.integers(-10000, 10000), n=st.integers(-10000, 10000)
)
def test_random_integers(m, n, a):
    shape = a.shape
    if m > n:
        m, n = n, m
    elif m == n:
        n = n+1
    b = random_integers(m, n, shape)
    assert b.shape == a.shape
    assert (m <= b.data).all() and (b.data <= n).all()


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
def test_random_sample(a):
    shape = a.shape
    b = random_sample(shape)
    assert b.shape == a.shape
    assert (0 <= b.data).all() and (b.data<= 1).all()


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
def test_ranf(a):
    shape = a.shape
    b = ranf(shape)
    assert b.shape == a.shape
    assert (0 <= b.data).all() and (b.data<= 1).all()


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    )
)
def test_sample(a):
    shape = a.shape
    b = sample(shape)
    assert b.shape == a.shape
    assert (0 <= b.data).all() and (b.data<= 1).all()
