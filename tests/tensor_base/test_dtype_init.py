import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
from tests.custom_strategies import tensors
from tests.utils import does_not_raise


def arrays_or_tensors(*args, **kwargs):
    return hnp.arrays(*args, **kwargs) | tensors(*args, **kwargs)


generic_data = st.sampled_from([0, 0.0, [0], [0.0]]) | arrays_or_tensors(
    dtype=hnp.floating_dtypes() | hnp.integer_dtypes(),
    shape=hnp.array_shapes(min_dims=0, min_side=0),
)

integer_data = st.sampled_from([0, [0]]) | arrays_or_tensors(
    dtype=hnp.integer_dtypes(),
    shape=hnp.array_shapes(min_dims=0, min_side=0),
)

float_data = st.sampled_from([0.0, [0.0]]) | arrays_or_tensors(
    dtype=hnp.floating_dtypes(),
    shape=hnp.array_shapes(min_dims=0, min_side=0),
)

integer_dtypes = st.sampled_from([int, "int32"]) | hnp.integer_dtypes()
float_dtypes = st.sampled_from([float, "float32"]) | hnp.floating_dtypes()


@given(
    data=st.data(),
    dtype=st.none() | integer_dtypes | hnp.boolean_dtypes(),
    constant=st.none() | st.booleans(),
    copy=st.booleans(),
    ndmin=st.integers(0, 3),
)
def test_integer_dtype_behavior(
    data: st.DataObject, dtype, constant, copy: bool, ndmin: int
):
    """Tests scenarios where tensor.dtype will ultimately be
    a integer dtype"""

    data_strat = integer_data if dtype is None else generic_data
    arr_data = data.draw(data_strat, label="arr_data")

    with (
        pytest.raises(ValueError) if constant is False else does_not_raise()
    ) as exec_info:
        # Setting `constant=False` for an integer-type tensor should raise
        tensor = mg.Tensor(
            arr_data, dtype=dtype, constant=constant, copy=copy, ndmin=ndmin
        )

    if exec_info is None:  # did not raise
        assert tensor.constant is True
        assert np.issubdtype(tensor.dtype, np.integer) or np.issubdtype(
            tensor.dtype, np.bool_
        )


@given(
    data=st.data(),
    dtype=st.none() | float_dtypes,
    constant=st.none() | st.booleans(),
    copy=st.booleans(),
    ndmin=st.integers(0, 3),
)
def test_float_dtype_behavior(
    data: st.DataObject, dtype, constant, copy: bool, ndmin: int
):
    """Tests scenarios where tensor.dtype will ultimately be
    a float dtype"""

    data_strat = float_data if dtype is None else generic_data
    arr_data = data.draw(data_strat, label="arr_data")
    tensor = mg.Tensor(arr_data, dtype=dtype, constant=constant, copy=copy, ndmin=ndmin)

    if constant is None:
        assert tensor.constant is False
    else:
        assert tensor.constant is constant

    assert np.issubdtype(tensor.dtype, np.floating)
