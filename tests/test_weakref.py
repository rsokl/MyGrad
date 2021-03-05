from typing import Callable
from weakref import WeakValueDictionary

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad import Tensor
from tests.custom_strategies import tensors


@pytest.mark.parametrize("func", [lambda x: +x, lambda x: x[...]], ids=["+x", "x[:]"])
@given(x=tensors())
def test_refs_that_point_forward_in_graph_are_weak(
    x: Tensor, func: Callable[[Tensor], Tensor]
):
    # op doesn't produce any references
    # thus `x` shouldn't record any ops
    func(x)
    assert all(i() is None for i in x._ops)
    assert len(x._view_children) == 0


@pytest.mark.parametrize("func", [lambda x: +x, lambda x: x[...]], ids=["+x", "x[:]"])
@given(x=tensors(constant=False, elements=st.floats(-10, 10)))
def test_derefrencing_tensor_from_upstream_in_graph_doesnt_break_graph(
    x: Tensor, func: Callable[[Tensor], Tensor]
):
    # op doesn't produce any references
    # thus `x` shouldn't record any ops
    y = func(x)
    z = 2 * y
    del y
    assert len(x._ops) == 1
    z.backward()
    assert_allclose(x.grad, 2 * np.ones_like(x))


@pytest.mark.parametrize("constant", [True, False])
def test_no_ref_op_does_not_prevent_gc(constant: bool):
    x = mg.arange(10.0, constant=constant)
    y = np.arange(10.0)
    refs = WeakValueDictionary({0: y})

    # participating in op should not prevent
    # y from being garbage collected
    y * x

    assert len(refs) == 1
    del y
    assert len(refs) == 0
