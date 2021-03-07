import numpy as np
from numpy.testing import assert_array_equal

import mygrad as mg


def test_allclose():
    assert np.allclose(mg.tensor([1.0, 2.0]), np.array([1.0, 2.0])) is True
    assert np.allclose(mg.tensor([1.0, 2.0]), np.array([1.0, 1.0])) is False


def test_bincount():
    w = mg.tensor([0.3, 0.5, 0.2, 0.7, 1.0, -0.6])  # weights
    x = mg.tensor([0, 1, 1, 2, 2, 2])
    assert_array_equal(np.bincount(x, weights=w), [0.3, 0.7, 1.1])


def test_can_cast():
    assert np.can_cast(mg.tensor(1000.0), np.float32) is True
    assert np.can_cast(mg.tensor([1000.0]), np.float32) is False


def test_may_share_memory():
    assert np.may_share_memory(mg.tensor([1, 2]), mg.tensor([5, 8, 9])) is False

    x = mg.zeros([3, 4])
    assert np.may_share_memory(x[:, 0], x[:, 1]) is True


def test_shares_memory():
    x = mg.tensor([1, 2, 3, 4])
    assert np.shares_memory(x, mg.tensor([5, 6, 7])) is False
    assert np.shares_memory(x[::2], x) is True
    assert np.shares_memory(x[::2], x[1::2]) is False


def test_result_type():
    assert np.result_type(mg.tensor(3.0), mg.tensor(-2)) is np.dtype("float64")


def test_min_scalar_type():
    assert np.min_scalar_type(mg.tensor(3.1)) is np.dtype("float16")
    assert np.min_scalar_type(mg.tensor(1e50)) is np.dtype("float64")
