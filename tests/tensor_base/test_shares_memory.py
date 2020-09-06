import numpy as np

import mygrad as mg


def test_shares_memory_works_with_tensors():
    x = mg.arange(10)
    y = x[:2]
    assert np.shares_memory(x, y)
    assert np.shares_memory(x.data, y)
    assert np.shares_memory(x, y.data)
    assert np.shares_memory(x.data, y.data)
    assert not np.shares_memory(x, mg.arange(10))
    assert not np.shares_memory(x.copy(), y)
    assert not np.shares_memory(x, y + 1)


def test_may_share_memory_works_with_tensors():
    x = mg.arange(10)
    y = x[:2]
    assert np.may_share_memory(x, y)
    assert np.may_share_memory(x.data, y)
    assert np.may_share_memory(x, y.data)
    assert np.may_share_memory(x.data, y.data)
    assert not np.may_share_memory(x, mg.arange(10))
    assert not np.may_share_memory(x.copy(), y)
    assert not np.may_share_memory(x, y + 1)


def test_tensor_base_matches_ndarray_base():
    tens = mg.arange(10)
    arr = np.arange(10)

    assert tens.base is None
    assert arr.base is None

    t1 = tens[:5]
    a1 = arr[:5]
    assert t1.base is tens
    assert t1.base.data is tens.data
    assert a1.base is arr

    t2 = t1[:2]
    a2 = a1[:2]

    assert t2.base is tens
    assert t2.data.base is tens.data
    assert a2.base is arr

    t3 = tens + 1
    a3 = arr + 1
    assert t3.base is None
    assert a3.base is None


def test_views_of_non_arrays_leave_no_base():
    assert mg.reshape(2.0, (1,)).base is None
    assert mg.reshape(list(range(9)), (3, 3)).base is None


def test_no_share_memory_view_is_still_view():
    # an empty array can be a view without sharing memory
    array = np.array([])
    array_view = array[tuple()]
    assert array_view.base is array, "expected numpy behavior does not hold"

    array_view_of_view = array_view[tuple()]
    assert (
        array_view_of_view.base is not array_view
    ), "expected numpy behavior does not hold"
    assert array_view_of_view.base is array, "expected numpy behavior does not hold"

    tensor = mg.Tensor([])
    tensor_view = tensor[tuple()]
    assert tensor_view.base is tensor

    tensor_view_of_view = tensor_view[tuple()]
    assert tensor_view_of_view.base is not tensor_view
    assert tensor_view_of_view.base is tensor
