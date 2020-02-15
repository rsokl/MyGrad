import numpy as np
from typing import Optional
import hypothesis.extra.numpy as hnp

from hypothesis import given
import hypothesis.strategies as st

from mygrad import Tensor
from numpy.testing import assert_array_equal


@given(
    tensor=st.tuples(
        hnp.arrays(
            shape=hnp.array_shapes(), dtype=hnp.integer_dtypes() | hnp.floating_dtypes()
        ),
        st.booleans(),
    ).map(lambda x: Tensor(x[0], constant=x[1])),
    dest_type=hnp.integer_dtypes() | hnp.floating_dtypes(),
    constant=st.booleans() | st.none(),
)
def test_astype(tensor: Tensor, dest_type: type, constant: Optional[bool]):
    tensor = tensor * 1  # give tensor a creator
    new_tensor = tensor.astype(dest_type, constant=constant)

    assert new_tensor.constant is (tensor.constant if constant is None else constant)
    assert tensor.creator is not None
    assert new_tensor.creator is None
    assert new_tensor.dtype is dest_type
    assert new_tensor.shape == tensor.shape

    if new_tensor.dtype is tensor.dtype:
        assert_array_equal(new_tensor.data, tensor.data)
