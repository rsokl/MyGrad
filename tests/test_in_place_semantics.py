from typing import Callable

import hypothesis.strategies as st
import pytest
from hypothesis import given
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import Tensor
from tests.custom_strategies import tensors


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
