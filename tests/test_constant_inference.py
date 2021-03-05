import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import mygrad as mg
from mygrad.math.arithmetic.ops import Multiply
from tests.custom_strategies import array_likes, valid_constant_arg
from tests.utils.checkers import expected_constant
from tests.utils.functools import SmartSignature
from tests.utils.wrappers import clears_mem_state


def mul(x, y, *, dtype=None, constant=None):
    return mg.Tensor._op(Multiply, x, y, op_kwargs=dict(dtype=dtype), constant=constant)


@clears_mem_state
@given(
    x=array_likes(shape=(1,), dtype=np.int32),
    y=array_likes(shape=(1,), dtype=np.int32),
    dtype=st.none() | st.just("float32") | st.just(int),
    data=st.data(),
)
def test_that_typical_op_propagates_constant_under_general_conditions(
    x, y, dtype, data: st.DataObject
):
    arr = np.multiply(x, y, dtype=dtype)
    constant = data.draw(valid_constant_arg(arr.dtype), label="constant")
    expected = expected_constant(x, y, dest_dtype=arr.dtype, constant=constant)

    out = mul(x, y, **SmartSignature(dtype=dtype, constant=constant).kwargs)
    assert out.constant is expected


def test_simple_constant_behavior():
    const = mg.tensor(1.0, constant=True)
    var = mg.tensor(1.0, constant=False)

    assert mg.add(const, const).constant is True
    assert mg.add(const, const, constant=False).constant is False
    assert mg.add(const, const, constant=True).constant is True

    assert mg.add(const, var).constant is False
    assert mg.add(const, var, constant=False).constant is False
    assert mg.add(const, var, constant=True).constant is True
