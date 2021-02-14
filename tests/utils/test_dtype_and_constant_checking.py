"""
Test utilities responsible for assuring that the `constant` attribute
of an output tensor is consistent with the dtype of the tensor, and is
inferred appropriately based on the input arguments to the function responsible
for creating the tensor.
"""

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

import mygrad as mg
from mygrad.typing import DTypeLikeReals
from tests.custom_strategies import real_dtypes
from tests.utils import InternalTestError, check_dtype_consistency, expected_constant


@given(dest_dtype=real_dtypes, as_tensor=st.booleans())
def test_dtype_consistency(dest_dtype: DTypeLikeReals, as_tensor: bool):

    out = (
        np.array(1, dtype=dest_dtype)
        if not as_tensor
        else mg.tensor(1, dtype=dest_dtype)
    )
    check_dtype_consistency(out, dest_dtype=dest_dtype)


@given(
    dtype1=real_dtypes.filter(lambda x: x is not None),
    dtype2=real_dtypes.filter(lambda x: x is not None),
)
def test_incompat_dtypes_raise(dtype1: DTypeLikeReals, dtype2: DTypeLikeReals):
    assume(np.dtype(dtype1) != np.dtype(dtype2))
    out = np.array(1, dtype=dtype1)

    with pytest.raises(AssertionError):
        check_dtype_consistency(out, dest_dtype=dtype2)


const_tensor = mg.tensor(1.0, constant=True)
var_tensor = mg.tensor(1.0, constant=False)


@given(dest_dtype=hnp.integer_dtypes())
def test_no_input_as_int_infers_constant_out(dest_dtype: DTypeLikeReals):
    assert expected_constant(dest_dtype=dest_dtype, constant=None) is True


@given(dest_dtype=hnp.floating_dtypes())
def test_no_input_as_float_infers_variable_out(dest_dtype: DTypeLikeReals):
    assert expected_constant(dest_dtype=dest_dtype, constant=None) is False


@given(
    inp=st.lists(
        st.sampled_from([const_tensor, np.array(1.0), var_tensor, 1, 1.0, [1]])
    ),
    dest_dtype=hnp.floating_dtypes(),
)
def test_any_variable_input_infers_to_variable_out(inp, dest_dtype):
    inp += [var_tensor]
    assert expected_constant(*inp, dest_dtype=dest_dtype, constant=None) is False


@given(
    inp=st.lists(st.sampled_from([const_tensor, np.array(1.0), 1, 1.0, [1]])),
    dest_dtype=real_dtypes.filter(lambda x: x is not None),
)
def test_all_constant_input_infers_to_constant_out(inp, dest_dtype):
    inp += [const_tensor]
    assert expected_constant(*inp, dest_dtype=dest_dtype, constant=None) is True


@given(
    inp=st.lists(
        st.sampled_from([const_tensor, np.array(1.0), var_tensor, 1, 1.0, [1]])
    ),
    dest_dtype=hnp.floating_dtypes(),
    constant=st.booleans(),
)
def test_specifying_constant_without_float_output_always_determines_output_state(
    inp, dest_dtype: DTypeLikeReals, constant: bool
):
    assert expected_constant(*inp, dest_dtype=dest_dtype, constant=constant) is constant


@given(
    inp=st.lists(
        st.sampled_from([const_tensor, np.array(1.0), var_tensor, 1, 1.0, [1]])
    ),
    dest_dtype=hnp.integer_dtypes(),
    constant=st.sampled_from([True, None]),
)
def test_integer_output_is_always_constant(
    inp, dest_dtype: DTypeLikeReals, constant: bool
):
    assert expected_constant(*inp, dest_dtype=dest_dtype, constant=constant) is True


@given(
    inp=st.lists(
        st.sampled_from([const_tensor, np.array(1.0), var_tensor, 1, 1.0, [1]])
    ),
    dest_dtype=hnp.integer_dtypes(),
)
def test_bad_constant_and_dtype_config_raises_internal_test_error(inp, dest_dtype):
    with pytest.raises(InternalTestError):
        expected_constant(*inp, dest_dtype=dest_dtype, constant=False)


@given(
    inp=st.lists(
        st.sampled_from([const_tensor, np.array(1.0), var_tensor, 1, 1.0, [1]])
    ),
    constant=st.none() | st.booleans(),
)
def test_specifying_none_for_dest_dtype_raises(inp, constant):
    with pytest.raises(InternalTestError):
        expected_constant(*inp, dest_dtype=None, constant=constant)
