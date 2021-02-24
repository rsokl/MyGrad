import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
from tests import ufuncs, populate_args
from tests.custom_strategies import tensors, no_value
from tests.utils import expected_constant
from tests.checkers import check_consistent_grad_dtype


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
        dtype=st.sampled_from(
            [
                np.float64,
                np.float32,
                int,
            ]
        ),
        shape=(1,),
        elements=st.just(1),
        constant=st.booleans(),
    )
    | st.just([1])
)


@pytest.mark.parametrize("ufunc", ufuncs)
def test_constant_propagates_correctly_according_to_dtype(ufunc):
    @given(
        inputs=st.tuples(*[simple_arr_likes] * ufunc.nin),
        constant=no_value() | st.none() | st.just(True),
        dtype=no_value() | st.just(np.float64),
    )
    def _runner(inputs, constant, dtype):
        args = populate_args(*inputs, constant=constant, dtype=dtype)
        out = ufunc(*args.args, **args.kwargs)

        _expected_constant = expected_constant(
            *args.args, constant=constant, dest_dtype=out.dtype
        )
        assert out.constant is _expected_constant

        out.backward()

        if out.constant is False:
            check_consistent_grad_dtype(out, *inputs)

    _runner()
