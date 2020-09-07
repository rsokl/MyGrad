import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from mygrad import Tensor
from tests.custom_strategies import tensors
from tests.utils import flags_to_dict


def test_memory_locks_during_graph():
    x = Tensor([1.0, 2.0])
    assert x.data.flags.writeable

    y = +x  # should lock memory
    with pytest.raises(ValueError):
        x.data *= 1
    y.backward()  # should release memory
    x.data *= 1


@given(tensor=tensors(shape=(2,), read_only=st.booleans(), constant=False),)
def test_lock_restore_writeability_roundtrip(tensor: Tensor,):
    original_flags = flags_to_dict(tensor)

    tensor._lock_writeability()

    assert not tensor.data.flags.writeable

    tensor._restore_writeability()

    restored_flags = flags_to_dict(tensor)

    assert tensor._data_was_writeable is None
    assert tensor._base_data_was_writeable is None
    assert original_flags == restored_flags


@given(read_only=st.booleans())
def test_lock_restore_writeability_with_base_roundtrip(read_only: bool):
    base_arr = np.array([1.0, 2.0])
    base_arr.flags.writeable = not read_only
    view = base_arr[...]

    original_flags = flags_to_dict(base_arr)
    view_flags = flags_to_dict(view)

    tensor = Tensor(view, _copy_data=False)

    tensor._lock_writeability()

    assert not base_arr.flags.writeable
    assert not view.flags.writeable

    tensor._restore_writeability()

    restored_flags = flags_to_dict(base_arr)
    restored_view_flags = flags_to_dict(view)

    assert tensor._data_was_writeable is None
    assert tensor._base_data_was_writeable is None
    assert original_flags == restored_flags
    assert view_flags == restored_view_flags


@given(tensor=tensors(shape=(2,), read_only=st.booleans(), constant=False),)
def test_lock_restore_writeability_roundtrip_of_view(tensor: Tensor):
    original_flags = flags_to_dict(tensor)

    view = tensor[...]

    view._lock_writeability()

    assert not tensor.data.flags.writeable
    assert not view.data.flags.writeable

    view._restore_writeability()

    restored_flags = flags_to_dict(tensor)

    assert tensor._data_was_writeable is None
    assert tensor._base_data_was_writeable is None
    assert original_flags == restored_flags

    assert view._data_was_writeable is None
    assert view._base_data_was_writeable is None
    assert view.data.flags.writeable is original_flags["WRITEABLE"]
