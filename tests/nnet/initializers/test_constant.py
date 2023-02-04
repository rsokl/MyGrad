import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given

from mygrad import Tensor
from mygrad.nnet.initializers import constant as constant_initializer
from tests.custom_strategies import valid_constant_arg


@given(
    array=hnp.arrays(
        dtype=hnp.integer_dtypes()
        | hnp.unsigned_integer_dtypes()
        | hnp.floating_dtypes(),
        shape=hnp.array_shapes(),
    ),
    data=st.data(),
)
def test_constant_initializer(array, data: st.DataObject):
    value = array.reshape(-1)[
        0
    ]  # let hypothesis do all the heavy lifting picking an acceptable value for the dtype
    assume(not np.isnan(value))
    constant = data.draw(valid_constant_arg(array.dtype), label="constant")
    tensor = constant_initializer(
        array.shape, value=value, dtype=array.dtype, constant=constant
    )

    assert isinstance(tensor, Tensor)
    assert np.allclose(tensor.data, value, equal_nan=True)
    assert tensor.data.shape == array.shape
    assert tensor.data.dtype == array.dtype
