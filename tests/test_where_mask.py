import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor
from mygrad.math.arithmetic.ops import Multiply
from mygrad.math.trigonometric.ops import Sin


def mul(x, y, *, constant=None, **kwargs) -> Tensor:
    return Tensor._op(Multiply, x, y, constant=constant, op_kwargs=kwargs)


def sin(x, *, constant=None, **kwargs) -> Tensor:
    return Tensor._op(Sin, x, constant=constant, op_kwargs=kwargs)


def test_manual_multiply_no_broadcast():
    x = Tensor([1.0, 2.0, 3.0])
    y = -x.copy()
    mask = np.array([True, False, True])
    out = mul(x, y, where=mask)
    assert_array_equal(out[mask], [-1.0, -9.0])
    assert not np.allclose(out[~mask], -4)
    out.backward([10.0, 1.0, 20.0])
    assert_array_equal(x.grad, [-10.0, 0.0, -60.0])
    assert_array_equal(y.grad, [10.0, 0.0, 60.0])


def test_manual_multiply_with_mask_broadcast():
    x = mg.arange(9.0).reshape(3, 3)
    y = -x.copy()
    mask = np.array([True, True, False])
    out = mul(x, y, where=mask)
    mask = np.broadcast_to(mask, (3, 3))
    assert_array_equal(out[mask].reshape(3, -1), -x.data[:, :2] ** 2)
    out.backward()
    assert_array_equal(x.grad, y.data * mask)
    assert_array_equal(y.grad, x.data * mask)


def test_manual_multiply_with_multiple_broadcast():
    x = mg.arange(9.0).reshape(3, 3)
    y = mg.tensor(-1.0)
    mask = np.array([True, True, False])
    out = mul(x, y, where=mask)
    mask = np.broadcast_to(mask, (3, 3))
    assert_array_equal(out[mask].reshape(3, -1), -x.data[:, :2])
    out.backward()
    assert_array_equal(
        x.grad, np.broadcast_to(y.data, x.shape) * np.broadcast_to(mask, x.shape)
    )
    assert_array_equal(y.grad, np.sum(x.data * mask))


def test_manual_sine_no_broadcast():
    x = mg.tensor([-mg.pi, 1.0, 0])
    mask = np.array([True, False, True])
    out = sin(x, where=mask)
    assert_allclose(out[mask], [-0, 0], atol=1e-7)
    assert not np.allclose(out[~mask], np.sin(1.0), atol=1e-7)
    out.backward([10.0, 1.0, 20.0])
    assert_array_equal(x.grad, [-10.0, 0.0, 20.0])
