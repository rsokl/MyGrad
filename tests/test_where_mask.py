from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor


def test_manual_multiply_no_broadcast():
    x = Tensor([1.0, 2.0, 3.0])
    y = -x.copy()
    mask = np.array([True, False, True])
    out = mg.multiply(x, y, where=mask)
    assert_array_equal(out[mask], [-1.0, -9.0])
    assert not np.allclose(out[~mask], -4)
    out.backward([10.0, 1.0, 20.0])
    assert_array_equal(x.grad, [-10.0, 0.0, -60.0])
    assert_array_equal(y.grad, [10.0, 0.0, 60.0])


def test_manual_multiply_with_mask_broadcast():
    x = mg.arange(9.0).reshape(3, 3)
    y = -x.copy()
    mask = np.array([True, True, False])
    out = mg.multiply(x, y, where=mask)
    mask = np.broadcast_to(mask, (3, 3))
    assert_array_equal(out[mask].reshape(3, -1), -x.data[:, :2] ** 2)
    out.backward()
    assert_array_equal(x.grad, y.data * mask)
    assert_array_equal(y.grad, x.data * mask)


def test_manual_multiply_with_multiple_broadcast():
    x = mg.arange(9.0).reshape(3, 3)
    y = mg.tensor(-1.0)
    mask = np.array([True, True, False])
    out = mg.multiply(x, y, where=mask)
    mask = np.broadcast_to(mask, (3, 3))
    assert_array_equal(out[mask].reshape(3, -1), -x.data[:, :2])
    out.backward()
    assert_array_equal(
        x.grad, np.broadcast_to(y.data, x.shape) * np.broadcast_to(mask, x.shape)
    )
    assert_array_equal(y.grad, np.sum(x.data * mask))


@pytest.mark.parametrize("process_x", [lambda x: x, lambda x: x[...], lambda x: +x])
@pytest.mark.parametrize("process_y", [lambda x: x, lambda x: x[...], lambda x: +x])
@pytest.mark.parametrize("process_out", [lambda x: x, lambda x: x[...], lambda x: +x])
def test_manual_multiply_with_out(
    process_x: Callable[[Tensor], Tensor],
    process_y: Callable[[Tensor], Tensor],
    process_out: Callable[[Tensor], Tensor],
):
    ox = Tensor([1.0, 2.0, 3.0])
    x = +ox
    y = -x.copy()
    mask = np.array([True, False, True])
    out = mg.multiply(process_x(x), process_y(y), where=mask, out=process_out(x))

    assert ox.data.flags.writeable is False
    assert out.data.flags.writeable is False

    (2 * out).backward()

    assert_allclose(ox.grad, [-2.0, 2.0, -6.0])
    assert_allclose(y.grad, [2.0, 0.0, 6.0])

    if out is x or out.base is x:
        assert_allclose(x.grad, [2.0, 2.0, 2.0])
    else:
        assert_allclose(x.grad, ox.grad)

    assert ox.data.flags.writeable is True
    assert out.data.flags.writeable is True


@pytest.mark.parametrize("process_x", [lambda x: x, lambda x: x[...], lambda x: +x])
@pytest.mark.parametrize("process_out", [lambda x: x, lambda x: x[...]])
def test_unary_func_with_where_and_out(process_x, process_out):
    ox = mg.tensor([1.0, 2.0, 3.0])
    x = +ox
    mask = np.array([True, False, True])
    mg.negative(process_x(x), where=mask, out=process_out(x))
    assert_allclose(x, [-1.0, 2.0, -3.0], atol=1e-7)
    x.backward([10.0, 2.0, 20.0])
    assert_array_equal(x.grad, [10.0, 2.0, 20.0])
    assert_array_equal(ox.grad, [-10.0, 2.0, -20.0])
