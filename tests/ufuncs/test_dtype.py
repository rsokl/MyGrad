import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
from tests.utils.functools import populate_args
from tests.utils.ufuncs import ufuncs
from tests.custom_strategies import tensors, no_value
from tests.utils.checkers import check_consistent_grad_dtype, expected_constant


@pytest.mark.parametrize("ufunc", ufuncs)
@pytest.mark.parametrize("dest_dtype", [np.float64, np.float32, np.float16])
def test_dtype_casts_correctly(ufunc, dest_dtype):
    if not np.can_cast(np.float16, dest_dtype):
        pytest.skip("invalid cast")

    x = mg.tensor(1.0, dtype=np.float16)
    args = [x] * ufunc.nin

    out = ufunc(*args, dtype=dest_dtype)

    assert out.dtype == dest_dtype
    out.backward()

    # tensor.grad.dtype should always match tensor.dtype
    check_consistent_grad_dtype(out, x)


simple_arr_likes = (
    tensors(
        dtype=st.sampled_from([np.float64, np.float32, int]),
        shape=st.sampled_from([(1,), (1, 1)]),
        elements=st.just(1),
        constant=st.booleans(),
    )
    | st.just([1])
)


@pytest.mark.parametrize("ufunc", ufuncs)
@given(
    constant=no_value() | st.none() | st.just(True),
    dtype=no_value() | st.just(np.float64),
    data=st.data(),
)
def test_constant_and_grad_propagates_correctly_according_to_dtype(
    ufunc, data: st.DataObject, constant, dtype
):
    inputs = data.draw(st.tuples(*[simple_arr_likes] * ufunc.nin), label="inputs")
    args = populate_args(*inputs, constant=constant, dtype=dtype)
    out = ufunc(*args.args, **args.kwargs)

    _expected_constant = expected_constant(
        *args.args, constant=constant, dest_dtype=out.dtype
    )
    assert out.constant is _expected_constant

    out.backward()

    if out.constant is False:
        check_consistent_grad_dtype(out, *inputs)
