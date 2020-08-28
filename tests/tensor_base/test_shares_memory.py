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
