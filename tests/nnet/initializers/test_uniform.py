import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

from mygrad import Tensor
from mygrad.nnet.initializers import uniform


@given(
    dtype=hnp.unsigned_integer_dtypes()
    | hnp.integer_dtypes()
    | hnp.complex_number_dtypes()
)
def test_uniform_dtype_validation(dtype):
    with pytest.raises(ValueError):
        uniform(0, 1, dtype=dtype)


_reasonable_floats = st.floats(-1000, 1000, width=32)
_bounds = (
    st.tuples(_reasonable_floats, _reasonable_floats)
    .filter(lambda x: x[0] != x[1])
    .map(sorted)
)


@given(data=st.data())
def test_uniform_value_validation(data):
    upper_bound = data.draw(_reasonable_floats)
    lower_bound = data.draw(st.floats(min_value=upper_bound))
    with pytest.raises(ValueError):
        uniform(10, lower_bound=lower_bound, upper_bound=upper_bound)


_large_shapes = ((100_000,), (100, 10, 100), (1000, 1, 100), (10, 10, 10, 10, 10))


@given(
    shape=st.sampled_from(_large_shapes),
    lower_bound=_reasonable_floats,
    upper_bound=_reasonable_floats,
)
def test_uniform_statistics(shape, lower_bound, upper_bound):
    # ensure a minimum interval width
    lower_bound, upper_bound = (
        min(lower_bound, upper_bound),
        max(lower_bound, upper_bound) + 1,
    )
    tensor = uniform(shape, lower_bound=lower_bound, upper_bound=upper_bound)
    assert isinstance(tensor, Tensor)
    assert tensor.min() >= lower_bound
    assert tensor.max() <= upper_bound

    hist, _ = np.histogram(tensor.data, bins=10)
    assert (max(hist) - min(hist)) / np.mean(hist) < 0.1


@given(
    shape=hnp.array_shapes(),
    bounds=_bounds,
    dtype=hnp.floating_dtypes(),
    constant=st.booleans(),
)
def test_uniform(shape, bounds, dtype, constant):
    tensor = uniform(
        shape,
        lower_bound=Tensor(bounds[0]),
        upper_bound=Tensor(bounds[1]),
        dtype=dtype,
        constant=constant,
    )
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.constant == constant
