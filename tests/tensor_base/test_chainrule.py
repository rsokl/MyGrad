from typing import Union

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.tensor_base import Tensor


def _check_grad(t: mg.Tensor, expr: Union[None, np.ndarray, float]):
    if t.constant:
        assert t.grad is None
    else:
        if expr is None:
            assert t.grad is None
        else:
            assert t.grad is not None
            assert_allclose(t.grad, expr)


def _check_cleared_node(t: mg.Tensor):
    assert not t._ops and not t._accum_ops and t.creator is None


def test_check_grad():
    # forced grad on constant
    x = Tensor(1.0, constant=True)
    x.grad = np.array(1.0)

    # bad: constant tensor should have grad None
    with pytest.raises(AssertionError):
        _check_grad(x, 1.0)

    x = Tensor(1.0, constant=False)
    x.backward()

    # bad: grad is 1. but expects None
    with pytest.raises(AssertionError):
        _check_grad(x, None)
    _check_grad(x, 1.0)

    x = Tensor(1.0, constant=False)

    # bad: grad is None but expects not-None
    with pytest.raises(AssertionError):
        _check_grad(x, 1.0)
    _check_grad(x, None)

    x = Tensor(1.0, constant=False)
    x.backward(2.0)

    # bad: grad is 1. but expects 2.
    with pytest.raises(AssertionError):
        _check_grad(x, 1.0)
    _check_grad(x, 2.0)


@given(
    x=st.floats(min_value=-1e-6, max_value=1e6),
    x_constant=st.booleans(),
    y=st.floats(min_value=-1e-6, max_value=1e6),
    y_constant=st.booleans(),
    z=st.floats(min_value=-1e-6, max_value=1e6),
    z_constant=st.booleans(),
    side_effects=st.booleans(),
)
def test_chainrule_scalar(
    x: float,
    x_constant: bool,
    y: float,
    y_constant: bool,
    z: float,
    z_constant: bool,
    side_effects,
):
    x = Tensor(x, constant=x_constant)
    y = Tensor(y, constant=y_constant)
    z = Tensor(z, constant=z_constant)

    f = mg.add(x * y, z)
    g = mg.add(x, z * f * f)

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

    assert f.constant is (x.constant and y.constant and z.constant)
    assert g.constant is (x.constant and z.constant and f.constant)

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
    r"""
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
    note(f"v4: {v4}")

    v4.backward(grad)

    # check fwd-pass produces reliable math
    assert v2.data == v1_val ** 2
    assert v3.data == np.exp(v1_val ** 2)
    assert v4.data == 2 * v3.data

    # check that constant propagates through graph reliably
    assert v1.constant is v1_const
    assert v2.constant is (v2_const or v1.constant)
    assert v3.constant is (v3_const or v2.constant or v1.constant)
    assert v4.constant is (v4_const or v3.constant or v2.constant or v1.constant)

    # check that gradients are correct
    # dL/d4
    _check_grad(v4, grad)
    # dL/d3 = dL/d4 * d4/d3
    _check_grad(v3, None if v4.constant else grad * 2)
    # dL/d2 = dL/d4 * d4/d3 * d3/d2
    _check_grad(
        v2, None if (v4.constant or v3.constant) else grad * (2 * np.exp(v2.data))
    )
    # dL/d2 = dL/d4 * d4/d3 * d3/d2 * d2/d1
    _check_grad(
        v1,
        None
        if (v4.constant or v3.constant or v2.constant)
        else grad * (2 * np.exp(v2.data)) * (2 * v1.data),
    )

    # check that backprop metadata cleared appropriately upon completion of backprop
    assert not v4._accum_ops
    assert not v3._accum_ops
    assert not v2._accum_ops
    assert not v1._accum_ops

    # check the backprop clears graph & clear graph always propagates through the graph
    _check_cleared_node(v4)
    _check_cleared_node(v3)
    _check_cleared_node(v2)
    _check_cleared_node(v1)


@pytest.mark.parametrize("v1_const", [True, False])
@pytest.mark.parametrize("v2_const", [True, False])
@pytest.mark.parametrize("v3_const", [True, False])
@pytest.mark.parametrize("v4_const", [True, False])
@pytest.mark.parametrize("v5_const", [True, False])
@given(
    v1_val=st.integers(-2, 2).map(float), grad=st.integers(-2, 2).map(float),
)
def test_fanout_graph(
    v1_val: float,
    v1_const: bool,
    v2_const: bool,
    v3_const: bool,
    v4_const: bool,
    v5_const: bool,
    grad: float,
):
    r"""
      v1
    / | \
   v2 v3 v4
    \ | /
     v5
    """
    v1 = Tensor(v1_val, constant=v1_const)
    v2 = mg.square(v1, constant=v2_const)
    v3 = mg.exp(v1, constant=v3_const)
    v4 = mg.multiply(v1, 2.0, constant=v4_const)
    v5 = mg.multiply_sequence(v2, v3, v4, constant=v5_const)

    note(f"v1: {v1}")
    note(f"v2: {v2}")
    note(f"v3: {v3}")
    note(f"v4: {v4}")
    note(f"v5: {v5}")

    v5.backward(grad)

    # check fwd-pass produces reliable math
    assert v2.data == v1_val ** 2
    assert v3.data == np.exp(v1_val)
    assert v4.data == 2 * v1_val
    assert v5.data == v2.data * v3.data * v4.data

    # check that constant propagates through graph reliably
    assert v1.constant is v1_const
    assert v2.constant is (v2_const or v1_const)
    assert v3.constant is (v3_const or v1_const)
    assert v4.constant is (v4_const or v1_const)
    assert v5.constant is v5_const or v1_const or (v2_const and v3_const and v4_const)

    # check that gradients are correct
    # dL/d5
    _check_grad(v5, grad)
    # dL/d4 = dL/d5 * p5/p4
    _check_grad(v4, None if v5_const else grad * (v2.data * v3.data))
    # dL/d3 = dL/d5 * p5/p3
    _check_grad(v3, None if v5_const else grad * (v2.data * v4.data))
    # dL/d2 = dL/d5 * p5/p2
    _check_grad(v2, None if v5_const else grad * (v3.data * v4.data))

    v1_grad = False
    # dL/d1 = +? dL/d2*p2/p1 +? dL/d3*p3/p1 +? dL/d4*p4/p1
    if not v5_const and not v2.constant:
        v1_grad += grad * (v3.data * v4.data) * 2 * v1_val

    if not v5_const and not v3.constant:
        v1_grad += grad * (v2.data * v4.data) * np.exp(v1_val)

    if not v5_const and not v4.constant:
        v1_grad += grad * (v2.data * v3.data) * 2

    if v1_grad is False:
        v1_grad = None

    _check_grad(
        v1, None if v5_const or (v2_const and v3_const and v4_const) else v1_grad
    )

    # check that backprop metadata cleared appropriately upon completion of backprop
    assert not v5._accum_ops
    assert not v4._accum_ops
    assert not v3._accum_ops
    assert not v2._accum_ops
    assert not v1._accum_ops

    # check the null grads & clear graph always propagates through the graph
    _check_cleared_node(v5)
    _check_cleared_node(v4)
    _check_cleared_node(v3)
    _check_cleared_node(v2)
    _check_cleared_node(v1)


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
    r"""
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

    # check fwd-pass produces reliable math
    assert v3.data == v1_val ** 2
    assert v4.data == (v2_val * v3.data)
    assert v5.data == (v4.data * v3.data)

    # check that constant propagates through graph reliably
    assert v1.constant is v1_const
    assert v2.constant is v2_const
    assert v3.constant is v3_const or v1.constant
    assert v4.constant is v4_const or (v2.constant and v3.constant)
    assert v5.constant is v5_const or (v3.constant and v4.constant)

    # check that gradients are correct
    # dL/d5
    _check_grad(v5, grad)

    # dL/d4 = dL/d5 * p5/p4
    _check_grad(v4, None if v5.constant else grad * v3.data)

    # dL/d3 = dL/d5 * p5/p3 +? dL/d4 * p4/p3
    v3_grad = (
        None
        if v5.constant
        else (grad * v4.data if v4.constant else grad * (2 * v3.data) * v2.data)
    )
    _check_grad(v3, v3_grad)

    # dL/d2 = dL/d4 * p4/p2
    v2_grad = None if (v5.constant or v4.constant) else grad * v3.data ** 2
    _check_grad(v2, v2_grad)

    # dL/d1 = dL/d3 * p3/p1
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

    # check that backprop metadata cleared appropriately upon completion of backprop
    assert not v5._accum_ops
    assert not v4._accum_ops
    assert not v3._accum_ops
    assert not v2._accum_ops
    assert not v1._accum_ops

    # check the null grads & clear graph always propagates through the graph
    _check_cleared_node(v5)
    _check_cleared_node(v4)
    _check_cleared_node(v3)
    _check_cleared_node(v2)
    _check_cleared_node(v1)


@pytest.mark.parametrize("v1_const", [True, False])
@pytest.mark.parametrize("v2_const", [True, False])
@pytest.mark.parametrize("v3_const", [True, False])
@pytest.mark.parametrize("v4_const", [True, False])
@pytest.mark.parametrize("v5_const", [True, False])
@given(
    v1_val=st.integers(-2, 2).map(float),
    v2_val=st.integers(-2, 2).map(float),
    grad=st.integers(-2, 2).map(float),
    # shrink to v5 (simplest pattern)
    dangling_site=st.integers(0, 4).map(lambda x: f"v{5 - x}"),
    dangling_const=st.booleans(),
)
def test_dynamic_interesting_graph(
    v1_val: float,
    v2_val: float,
    v1_const: bool,
    v2_const: bool,
    v3_const: bool,
    v4_const: bool,
    v5_const: bool,
    dangling_const: bool,
    dangling_site: str,
    grad: float,
):
    r"""
     --------v1------
     |       /\     |
     |      ---     |
     |       |      |
     |  v2   v3--   | .....
     |  |    |  |   |     |
     ---------  |   |    ---
           |    |   |     |
           v4   |   |  dangling
           |    |   |
           ----------
                |
                v5

    For each run of the test, v(1-5) is selected for the "dead_leaf"
    to branch from it (via the operation 3x(node) ). This dead leaf
    should have no effect on the rest of the graph - including backprop
    through it.
    """
    v1 = Tensor(v1_val, constant=v1_const)
    v2 = Tensor(v2_val, constant=v2_const)
    v3 = mg.multiply(v1, v1, constant=v3_const)
    v4 = mg.multiply_sequence(v1, v2, v3, constant=v4_const)
    v5 = mg.multiply_sequence(v1, v3, v4, constant=v5_const)

    dangling_site = locals()[dangling_site]  # type: Tensor
    dead_leaf = mg.multiply(dangling_site, 3.0, constant=dangling_const)
    v5.backward(grad)

    note(f"v1: {v1}")
    note(f"v2: {v2}")
    note(f"v3: {v3}")
    note(f"v4: {v4}")
    note(f"v5: {v5}")
    note(f"dead_leaf: {dead_leaf}")

    # check fwd-pass produces reliable math
    assert v3.data == v1_val ** 2
    assert v4.data == (v1_val * v2_val * v3.data)
    assert v5.data == (v4.data * v3.data * v1_val)
    assert dead_leaf.data == dangling_site.data * 3.0

    # check that constant propagates through graph reliably
    assert v1.constant is v1_const
    assert v2.constant is v2_const
    assert v3.constant is v3_const or v1.constant
    assert v4.constant is v4_const or (v1.constant and v2.constant and v3.constant)
    assert v5.constant is v5_const or (v1.constant and v3.constant and v4.constant)
    assert dead_leaf.constant is dangling_const or dangling_site.constant

    # check that gradients are correct
    _check_grad(v5, grad)

    # dL/d4 = dL/d5 * p5/p4
    _check_grad(v4, None if v5.constant else grad * v3.data * v1.data)

    v3_grad = (
        None
        if v5.constant
        else grad
        * v1.data
        * (v4.data if v4.constant else (2 * v3.data) * v2.data * v1.data)
    )
    _check_grad(v3, v3_grad)

    v2_grad = (
        None if (v5.constant or v4.constant) else grad * v3.data ** 2 * v1.data ** 2
    )
    _check_grad(v2, v2_grad)

    v1_grad = None if v5.constant else grad * v4.data * v3.data
    if not v5.constant and not v4.constant:
        v1_grad += v4.grad * v2.data * v3.data

    if not v5.constant and not v3.constant:
        v1_grad += 2 * v3.grad * v1_val

    _check_grad(v1, v1_grad)
    _check_grad(dead_leaf, None)

    # check that backprop metadata cleared appropriately upon completion of backprop
    assert not v5._accum_ops
    assert not v4._accum_ops
    assert not v3._accum_ops
    assert not v2._accum_ops
    assert not v1._accum_ops
    assert not dead_leaf._accum_ops and dead_leaf.creator is not None

    # check the null grads & clear graph always propagates through the graph
    _check_cleared_node(v5)
    _check_cleared_node(v4)
    _check_cleared_node(v3)
    _check_cleared_node(v2)
    _check_cleared_node(v1)
    assert dead_leaf.creator is not None
