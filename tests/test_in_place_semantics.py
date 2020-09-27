from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import Tensor


@pytest.mark.parametrize("constant", [True, False])
def test_in_place_op_propagates_to_views(constant: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y = +x

    view1 = y[...]
    view2 = view1[...]  # view-of-view
    y[:2] = -1  # should mutate all views
    assert y.base is None
    assert view1.base is y
    assert view2.base is y
    assert_array_equal(x, mg.arange(1.0, 5.0))

    assert_array_equal(y, [-1.0, -1.0, 3.0, 4.0])
    assert_array_equal(y, view1)
    assert_array_equal(y, view2)


@pytest.mark.parametrize(
    "identity_op",
    [
        lambda x: x,  # backprop directly post-setitem var
        lambda x: +x,  # backprop from downstream node
        lambda x: x[...],  # backprop from downstream view
    ],
)
def test_writing_a_view_with_a_view(identity_op: Callable[[Tensor], Tensor]):
    x = mg.arange(1.0, 5.0)
    y = +x
    dangling_view = y[...]
    y[:2] = y[-2:]  # y = [3, 4, 3, 4]
    proxy_y = identity_op(y)
    # -1 x2 + 2 x3 + -3 x2 + 4 x3 -> -4 x2 + 6 x3
    ([-1, 2, -3, 4] * proxy_y).sum().backward()
    assert_array_equal(proxy_y.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(y.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(x.grad, [0.0, 0.0, -4.0, 6.0])

    assert_array_equal(y, dangling_view)
    assert dangling_view.base is y
    assert dangling_view.grad is None
