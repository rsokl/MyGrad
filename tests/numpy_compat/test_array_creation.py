import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given

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
