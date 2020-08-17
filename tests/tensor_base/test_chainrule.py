import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.tensor_base import Tensor


def _check_grad(t, expr):
    if t.constant:
        assert t.grad is None
    else:
        if expr is None:
            assert t.grad is None
        else:
            assert t.grad is not None
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


@given(st.booleans())
def test_identical_inputs(constant):
    v1 = Tensor(2.0, constant=constant)
    v2 = v1 + v1
    v3 = v2 + v2
    v3.backward(1.0)  # v3 = 4 * v1
    assert v3.data.item() == 8.0
    _check_grad(v1, 4.0)


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


@pytest.mark.parametrize("v1_const", [True, False])
@pytest.mark.parametrize("v2_const", [True, False])
@pytest.mark.parametrize("v3_const", [True, False])
@pytest.mark.parametrize("v4_const", [True, False])
@given(
    v1_val=st.integers(-2, 2).map(float), grad=st.integers(-2, 2).map(float),
)
def test_linear_graph(
    v1_val: float,
    v1_const: bool,
    v2_const: bool,
    v3_const: bool,
    v4_const: bool,
    grad: float,
):
    """
     v1
     |
     v2
     |
     v3
     |
     v4
    """
    v1 = Tensor(v1_val, constant=v1_const)
    v2 = mg.square(v1, constant=v2_const)
    v3 = mg.exp(v2, constant=v3_const)
    v4 = mg.multiply(v3, 2.0, constant=v4_const)

    note(f"v1: {v1}")
    note(f"v2: {v2}")
    note(f"v3: {v3}")
    note(f"v3: {v4}")

    v4.backward(grad)

    assert v2.data == v1_val ** 2
    assert v3.data == np.exp(v1_val ** 2)
    assert v4.data == 2 * v3.data

    assert v1.constant is v1_const
    assert v2.constant is (v2_const or v1.constant)
    assert v3.constant is (v3_const or v2.constant or v1.constant)
    assert v4.constant is (v4_const or v3.constant or v2.constant or v1.constant)

    _check_grad(v4, None if v4.constant else grad)
    _check_grad(v3, None if v4.constant else 2 * grad)
    _check_grad(
        v2, None if (v4.constant or v3.constant) else 2 * grad * np.exp(v2.data)
    )
    _check_grad(
        v1,
        None
        if (v4.constant or v3.constant or v2.constant)
        else 4 * grad * v1_val * np.exp(v2.data),
    )

    assert not v3._accum_ops and v3.creator is not None
    assert not v2._accum_ops and v2.creator is not None
    assert not v1._accum_ops and v1.creator is None

    v4.null_gradients(clear_graph=True)
    assert v4.grad is None and v4.creator is None
    assert v3.grad is None and not v3._ops and v3.creator is None
    assert v2.grad is None and not v2._ops and v2.creator is None
    assert v1.grad is None and not v1._ops and v1.creator is None


@pytest.mark.parametrize("v1_const", [True, False])
@pytest.mark.parametrize("v2_const", [True, False])
@pytest.mark.parametrize("v3_const", [True, False])
@pytest.mark.parametrize("v4_const", [True, False])
@pytest.mark.parametrize("v5_const", [True, False])
@given(
    v1_val=st.integers(-2, 2).map(float),
    v2_val=st.integers(-2, 2).map(float),
    grad=st.integers(-2, 2).map(float),
)
def test_interesting_graph(
    v1_val: float,
    v2_val: float,
    v1_const: bool,
    v2_const: bool,
    v3_const: bool,
    v4_const: bool,
    v5_const: bool,
    grad: float,
):
    """
             v1
             /\
            ---
             |
        v2   v3--
        |    |  |
        ------  |
           |    |
           v4   |
           |    |
           ------
              |
              v5
    """
    v1 = Tensor(v1_val, constant=v1_const)
    v2 = Tensor(v2_val, constant=v2_const)
    v3 = mg.multiply(v1, v1, constant=v3_const)
    v4 = mg.multiply(v2, v3, constant=v4_const)
    v5 = mg.multiply(v3, v4, constant=v5_const)
    v5.backward(grad)

    note(f"v1: {v1}")
    note(f"v2: {v2}")
    note(f"v3: {v3}")
    note(f"v4: {v4}")
    note(f"v5: {v5}")

    assert v1.constant is v1_const
    assert v2.constant is v2_const
    assert v3.constant is v3_const or v1.constant
    assert v4.constant is v4_const or (v2.constant and v3.constant)
    assert v5.constant is v5_const or (v3.constant and v4.constant)

    assert v3.data == v1_val ** 2
    assert v4.data == (v2_val * v3.data)
    assert v5.data == (v4.data * v3.data)

    _check_grad(v5, None if v5.constant else grad)
    _check_grad(v4, None if v5.constant else grad * v3.data)

    v3_grad = (
        None
        if v5.constant
        else (grad * v4.data if v4.constant else 2 * grad * v2.data * v3.data)
    )
    _check_grad(v3, v3_grad)

    v2_grad = None if (v5.constant or v4.constant) else grad * v3.data ** 2
    _check_grad(v2, v2_grad)

    v1_grad = (
        None
        if (v5.constant or v3.constant)
        else (
            2 * grad * v1.data * v4.data
            if v4.constant
            else grad * 4 * v1.data ** 3 * v2.data
        )
    )
    _check_grad(v1, v1_grad)

    assert not v5._accum_ops and v5.creator is not None
    assert not v4._accum_ops and v4.creator is not None
    assert not v3._accum_ops and v3.creator is not None
    assert not v2._accum_ops and v2.creator is None
    assert not v1._accum_ops and v1.creator is None

    v5.null_gradients(clear_graph=True)
    assert v5.grad is None and v5.creator is None
    assert v4.grad is None and not v4._ops and v4.creator is None
    assert v3.grad is None and not v3._ops and v3.creator is None
    assert v2.grad is None and not v2._ops and v2.creator is None
    assert v1.grad is None and not v1._ops and v1.creator is None
