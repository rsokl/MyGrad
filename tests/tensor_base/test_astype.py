import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_array_equal

from mygrad import Tensor
from tests.custom_strategies import tensors, valid_constant_arg

real_types = (
    hnp.integer_dtypes() | hnp.unsigned_integer_dtypes() | hnp.floating_dtypes()
)


@given(
    tensor=tensors(dtype=real_types),
    dest_type=real_types,
    data=st.data(),
)
def test_astype(tensor: Tensor, dest_type: np.dtype, data: st.DataObject):
    tensor = +tensor  # give tensor a creator
    constant = data.draw(valid_constant_arg(dest_type), label="constant")
    new_tensor = tensor.astype(dest_type, constant=constant)

    expected_tensor = Tensor(tensor, dtype=dest_type, constant=constant)

    assert new_tensor.constant is expected_tensor.constant
    assert tensor.creator is not None
    assert new_tensor.creator is None
    assert new_tensor.dtype == dest_type
    assert new_tensor.shape == tensor.shape
    assert new_tensor.data is not tensor.data
    assert_array_equal(new_tensor.data, expected_tensor.data)


@settings(max_examples=30)
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
