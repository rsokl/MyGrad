import numpy as np
import pytest
from numpy.testing import assert_allclose

import mygrad as mg


# TODO: These tests need to be modified to check that tensor-identity
# is preserved by the augmented update once #280 is resolved
# https://github.com/rsokl/MyGrad/pull/280/
def test_augmented_multiply():
    a = np.arange(9.0).reshape(3, 3)
    expected_grad = np.where(a < 4, 2, 1)
    a[a < 4] *= 2

    t1 = mg.arange(9.0).reshape(3, 3)
    y = mg.full(t1[t1 < 4].shape, 2.0)
    # creates a new node in the graph so that `t1` can
    # effectively mirror `t2` before it was mutated, and
    # thus store the gradient downstream of the augmented update
    t2 = +t1
    t2[t2 < 4] *= y
    t2.sum().backward()

    assert_allclose(desired=a, actual=t2.data)
    assert_allclose(desired=expected_grad, actual=t1.grad)
    assert_allclose(desired=np.arange(4.0), actual=y.grad)

    assert not t2._ops and not t2._accum_ops
    assert not t1._ops and not t1._accum_ops
    assert not y._ops and not y._accum_ops


def test_augmented_add():
    a = np.arange(9.0).reshape(3, 3)
    expected_grad = np.where(a < 4, 1, 1)
    a[a < 4] += 2

    t1 = mg.arange(9.0).reshape(3, 3)
    y = mg.full(t1[t1 < 4].shape, 2.0)
    t2 = +t1
    t2[t2 < 4] += y
    t2.sum().backward()

    assert_allclose(desired=a, actual=t2.data)
    assert_allclose(desired=expected_grad, actual=t1.grad)
    assert_allclose(desired=np.ones((4,), dtype=float), actual=y.grad)

    assert not t2._ops and not t2._accum_ops
    assert not t1._ops and not t1._accum_ops
    assert not y._ops and not y._accum_ops


def test_augmented_divide():
    a = np.arange(9.0).reshape(3, 3)
    expected_grad = np.where(a < 4, 1 / 2, 1)
    a[a < 4] /= 2

    t1 = mg.arange(9.0).reshape(3, 3)
    y = mg.full(t1[t1 < 4].shape, 2.0)
    t2 = +t1
    t2[t2 < 4] /= y
    t2.sum().backward()

    assert_allclose(desired=a, actual=t2.data)
    assert_allclose(desired=expected_grad, actual=t1.grad)
    assert_allclose(desired=-np.arange(4.0) / 4, actual=y.grad)

    assert not t2._ops and not t2._accum_ops
    assert not t1._ops and not t1._accum_ops
    assert not y._ops and not y._accum_ops


def test_augmented_subtract():
    a = np.arange(9.0).reshape(3, 3)
    expected_grad = np.where(a < 4, 1, 1)
    a[a < 4] -= 2

    t1 = mg.arange(9.0).reshape(3, 3)
    y = mg.full(t1[t1 < 4].shape, 2.0)
    t2 = +t1
    t2[t2 < 4] -= y
    t2.sum().backward()

    assert_allclose(desired=a, actual=t2.data)
    assert_allclose(desired=expected_grad, actual=t1.grad)
    assert_allclose(desired=-np.ones((4,), dtype=float), actual=y.grad)
    assert not t2._ops and not t2._accum_ops
    assert not t1._ops and not t1._accum_ops
    assert not y._ops and not y._accum_ops


def test_augmented_power():
    a = np.arange(9.0).reshape(3, 3)
    expected_grad = np.where(a < 4, 2 * a, 1)
    a[a < 4] **= 2

    t1 = mg.arange(9.0).reshape(3, 3)
    y = mg.full(t1[t1 < 4].shape, 2.0)
    t2 = +t1
    t2[t2 < 4] **= y
    t2.sum().backward()

    assert_allclose(desired=a, actual=t2.data)
    assert_allclose(desired=expected_grad, actual=t1.grad)
    assert_allclose(
        desired=np.arange(4.0) ** 2 * np.log([1.0, 1.0, 2.0, 3.0]), actual=y.grad
    )

    assert not t2._ops and not t2._accum_ops
    assert not t1._ops and not t1._accum_ops
    assert not y._ops and not y._accum_ops


def test_augmented_matmul():
    a = np.arange(9.0).reshape(3, 3)
    t = mg.arange(9.0).reshape(3, 3)

    try:
        a[a < 4] @= np.arange(4.0)
    except Exception as e:
        with pytest.raises(type(e), match="Use 'a = a @ b' instead of 'a @= b'"):
            t[t < 4] @= mg.arange(4.0)
