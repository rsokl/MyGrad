import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_array_equal

import mygrad as mg
from tests.custom_strategies import tensors


@given(
    tensor=tensors(dtype=hnp.floating_dtypes(), shape=hnp.array_shapes()),
    as_array=st.booleans(),
    set_dtype=st.booleans(),
    set_order=st.booleans(),
)
def test_asarray_no_copy(
    tensor: mg.Tensor, as_array: bool, set_dtype: bool, set_order: bool
):
    """Ensures that asarray, called with compatible dtype/order returns a reference
    to the array"""
    if as_array:
        tensor = tensor.data

    kwargs = {}

    if set_dtype:
        kwargs["dtype"] = tensor.dtype
    if set_order:
        kwargs["order"] = "C"
    array = mg.asarray(tensor, **kwargs)
    assert array is (tensor.data if not as_array else tensor)


def _list(x: np.ndarray):
    return x.tolist()


@given(
    array_data=hnp.arrays(dtype=np.int8, shape=hnp.array_shapes()),
    convert_input=st.sampled_from([mg.Tensor, lambda x: x, _list]),
    dtype=hnp.floating_dtypes() | hnp.integer_dtypes() | st.none(),
    order=st.sampled_from(["C", "F", None]),
)
def test_asarray_returns_array_with_expected_data_and_attributes(
    array_data, convert_input, dtype, order
):
    data = convert_input(array_data)

    actual = mg.asarray(data, dtype=dtype, order=order)
    expected = np.asarray(data, dtype=dtype, order=order)

    assert isinstance(actual, np.ndarray)
    assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape

    if order is None:
        order = "C"

    assert actual.flags[f"{order.capitalize()}_CONTIGUOUS"]

# Test the third example from
# https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
def test_asarray_copies_consistently():
    a = mg.Tensor([1, 2], dtype=np.float32)

    assert np.asarray(a, dtype=np.float32) is a.data
    assert np.asarray(a, dtype=np.float64) is not a.data

    assert mg.asarray(a, dtype=np.float32) is a.data
    assert mg.asarray(a, dtype=np.float64) is not a.data
