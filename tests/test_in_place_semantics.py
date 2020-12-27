from typing import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor
from tests.custom_strategies import tensors


def test_raising_during_in_place_op_doesnt_corrupt_graph():
    x = mg.arange(1.0, 5.0)
    y = 2 * x
    w = y[...]
    with pytest.raises(ValueError):
        y[:2] = y  # shape mismatch

    (2 * w).backward()
    assert y.base is None
    assert w.base is y
    assert np.shares_memory(w, y)
    assert_allclose(w.grad, 2 * np.ones_like(y))
    assert_allclose(y.grad, 2 * np.ones_like(y))
    assert_allclose(x.grad, 4 * np.ones_like(y))


@pytest.mark.parametrize("constant", [True, False])
def test_in_place_op_propagates_to_views(constant: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y = +x

    view1 = y[...]
    view2 = view1[...]  # view of view
    y[:2] = -1  # should mutate all views
    assert y.base is None
    assert view1.base is y
    assert view2.base is y
    assert_array_equal(x, mg.arange(1.0, 5.0))

    assert_array_equal(y, [-1.0, -1.0, 3.0, 4.0])
    assert_array_equal(y, view1)
    assert_array_equal(y, view2)


@given(tensors(shape=(4,), constant=False))
def test_simple_backprop_from_view_post_upstream_mutation(x: Tensor):
    y = +x
    z = y[...]
    y[:2] = 0  # base is mutated
    # downstream view should carry appropriate info
    # for backprop post-mutation
    z.backward()

    assert_array_equal(z, y)
    assert_array_equal(z.grad, np.ones_like(y))
    assert_array_equal(y.grad, np.ones_like(y))
    assert_array_equal(x.grad, [0.0, 0.0, 1.0, 1.0])


@given(tensors(shape=(4,), elements=st.floats(-10, 10), constant=False))
def test_mutation_doesnt_corrupt_upstream_op(x: Tensor):
    y = +x
    view = y[...]

    # z = x**4
    z = mg.multiply_sequence(x, y, view, view[...])

    y[:2] = 0  # shouldn't change backprop through z

    z.backward()  # dz/dx = 6 * x ** 2

    assert_allclose(z, x.data ** 4)
    assert_array_equal(view, y)
    assert_allclose(z.grad, np.ones_like(y))
    assert_allclose(x.grad, 4 * np.asarray(x) ** 3)

    assert y.grad is None
    assert view.grad is None


@pytest.mark.parametrize("constant", [True, False])
def test_sets_and_restores_writeability(constant: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y = +x
    y[...] = 0
    assert x.data.flags.writeable is False
    assert y.data.flags.writeable is False
    y.backward()
    assert x.data.flags.writeable is True
    assert y.data.flags.writeable is True


@pytest.mark.parametrize("as_view", [True, False])
@given(x=tensors(read_only=True))
def test_respects_original_writeability(x: Tensor, as_view: bool):
    assert x.data.flags.writeable is False
    if as_view:
        x = x[...]

    with pytest.raises(ValueError):
        x[...] = 0


@pytest.mark.parametrize("constant", [True, False])
def test_respects_disabled_memguard(constant: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y = +x
    with mg.mem_guard_off:
        y[...] = 0
    assert x.data.flags.writeable is False
    assert y.data.flags.writeable is True
    y.backward()
    assert x.data.flags.writeable is True
    assert y.data.flags.writeable is True


@pytest.mark.parametrize(
    "target_op",
    [
        lambda x: x,  # backprop directly post-setitem var
        lambda x: +x,  # backprop from downstream node
        lambda x: x[...],  # backprop from downstream view
    ],
)
@pytest.mark.parametrize(
    "source_op",
    [
        lambda x: x,  # direct view
        lambda x: +x,  # downstream of view
        lambda x: x[...],  # view of view
    ],
)
@given(num_in_place_updates=st.integers(1, 3))
def test_writing_a_view_with_a_view(
    target_op: Callable[[Tensor], Tensor],
    source_op: Callable[[Tensor], Tensor],
    num_in_place_updates: int,
):
    x = mg.arange(1.0, 5.0)
    y = +x
    dangling_view = y[...]

    for _ in range(num_in_place_updates):
        # after the first in-place update, any additional
        # should have no further effect
        y[:2] = source_op(y[-2:])  # y = [3, 4, 3, 4]

    proxy_y = target_op(y)

    # output: -1 x2 + 2 x3 + -3 x2 + 4 x3 -> -4 x2 + 6 x3
    ([-1, 2, -3, 4] * proxy_y).sum().backward()

    assert_array_equal(proxy_y.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(y.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(x.grad, [0.0, 0.0, -4.0, 6.0])

    assert_array_equal(y, dangling_view)
    assert dangling_view.base is y
    assert dangling_view.grad is None

    dangling_view.clear_graph()  # release memory

    assert x.data.flags.writeable
    assert y.data.flags.writeable
    assert dangling_view.data.flags.writeable


def test_setitem_preserves_view_children():
    x = mg.arange(10.0)
    y = x[...]
    (view_child,) = x._view_children
    assert view_child is y

    x[...] = y
    (view_child,) = x._view_children
    assert view_child is y
