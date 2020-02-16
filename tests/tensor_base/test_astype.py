from typing import Optional

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_array_equal

from mygrad import Tensor

real_types = (
    hnp.integer_dtypes() | hnp.unsigned_integer_dtypes() | hnp.floating_dtypes()
)


@given(
    tensor=st.tuples(
        hnp.arrays(shape=hnp.array_shapes(), dtype=real_types), st.booleans(),
    ).map(lambda x: Tensor(x[0], constant=x[1])),
    dest_type=real_types,
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


@pytest.mark.parametrize(
    "type_strategy",
    [hnp.integer_dtypes(), hnp.unsigned_integer_dtypes(), hnp.floating_dtypes()],
)
@given(data=st.data())
def test_upcast_roundtrip(type_strategy, data: st.DataObject):
    thin, wide = data.draw(
        st.tuples(type_strategy, type_strategy).map(
            lambda x: sorted(x, key=lambda y: np.dtype(y).itemsize)
        )
    )
    orig_tensor = data.draw(
        hnp.arrays(
            dtype=thin,
            shape=hnp.array_shapes(),
            elements=hnp.from_dtype(thin).filter(np.isfinite),
        ).map(Tensor)
    )

    roundtripped_tensor = orig_tensor.astype(wide).astype(thin)
    assert_array_equal(orig_tensor, roundtripped_tensor)
