import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.tensor_base import Tensor


def _check_grad(t, expr):
    if t.constant:
        assert t.grad is None
    else:
        assert_allclose(t.grad, expr)


@given(
    x=st.floats(min_value=-1e-6, max_value=1e6),
    x_constant=st.booleans(),
    y=st.floats(min_value=-1e-6, max_value=1e6),
    y_constant=st.booleans(),
    z=st.floats(min_value=-1e-6, max_value=1e6),
    z_constant=st.booleans(),
    f_constant=st.just(True),
    g_constant=st.just(True),
    side_effects=st.booleans(),
)
def test_chainrule_scalar(
    x: float,
    x_constant: bool,
    y: float,
    y_constant: bool,
    z: float,
    z_constant: bool,
    f_constant: bool,
    g_constant: bool,
    side_effects,
):
    x = Tensor(x, constant=x_constant)
    y = Tensor(y, constant=y_constant)
    z = Tensor(z, constant=z_constant)

    f = mg.add(x * y, z, constant=f_constant)
    g = mg.add(x, z * f * f, constant=g_constant)

    if side_effects:
        # check side effects
        unused = 2 * g - f
        w = 1 * f
    else:
        unused = Tensor(0)
        w = Tensor(0)
    assert unused is not None

    assert x.constant is x_constant
    assert y.constant is y_constant
    assert z.constant is z_constant

    assert f.constant is f_constant or (x.constant and y.constant and z.constant)
    assert g.constant is g_constant or (x.constant and z.constant and f.constant)

    g.backward()

    assert w.grad is None

    if g.constant:
        assert g.grad is None
        assert f.grad is None
        assert x.grad is None
        assert y.grad is None
        assert z.grad is None
        return
    _check_grad(f, 2 * z.data * f.data)

    _check_grad(x, 1 + 2 * z.data * f.data * y.data)
    _check_grad(y, 2 * z.data * f.data * x.data)
    _check_grad(z, f.data ** 2 + z.data * 2 * f.data)


def test_identical_inputs():
    v1 = Tensor(2.0, constant=False)
    v2 = v1 + v1
    v3 = v2 + v2
    v3.backward(1.0)  # v3 = 4 * v1
    assert v3.data.item() == 8.0
    assert v1.grad.item() == 4.0


@given(data=st.floats(-10, 10), grad=(st.none() | st.floats(-10, 10)))
def test_non_broadcastable(data, grad):
    v1 = Tensor(data, constant=False)
    v2 = mg.exp(v1)
    v3 = mg.cos(v2)
    v3.backward(grad)

    if grad is None:
        grad = 1.0

    assert_allclose(actual=v2.data, desired=np.exp(v1.data))
    assert_allclose(actual=v3.data, desired=np.cos(np.exp(v1.data)))

    assert_allclose(actual=v3.grad, desired=grad)
    assert_allclose(actual=v2.grad, desired=-np.sin(v2.data) * grad)
    assert_allclose(actual=v1.grad, desired=np.exp(v1.data) * -np.sin(v2.data) * grad)
