from typing import Callable
from weakref import WeakValueDictionary

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
    assert len(x._ops) == 0
    assert len(x._view_children) == 0


@pytest.mark.parametrize("func", [lambda x: +x, lambda x: x[...]], ids=["+x", "x[:]"])
@given(x=tensors(constant=False))
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


@pytest.mark.parametrize("constant", [True, False])
def test_no_ref_op_does_not_lock_data(constant: bool):
    x = mg.arange(10.0, constant=constant)
    y = np.arange(10.0)
    assert y.flags.writeable is True

    # participating in unreferenced op should not
    # lock y's writeability
    y * x

    assert y.flags.writeable is True


@pytest.mark.parametrize("func", [lambda x: +x, lambda x: x[...]], ids=["+x", "x[:]"])
@pytest.mark.parametrize("constant", [True, False])
def test_dereferencing_tensor_restores_data_writeability(
    constant: bool, func: Callable[[Tensor], Tensor]
):
    x = mg.arange(2.0, constant=constant)
    data = x.data

    y = +x

    assert data.flags.writeable is False
    del y
    assert data.flags.writeable is False, (
        "de-referencing down-stream tensor should "
        "not change writeability of up-stream data"
    )
    del x
    assert data.flags.writeable is True


@pytest.mark.parametrize("constant", [True, False])
def test_only_final_dereference_restores_writeability(constant: bool):
    x = mg.arange(10.0, constant=constant)
    y = np.arange(10.0)
    assert y.flags.writeable is True

    w = y * x
    z = y * x

    del w
    assert y.flags.writeable is False, (
        "`y` is still involved with `z` and `x`; "
        "its writeability should not be restored"
    )
    del z
    assert y.flags.writeable is True


def test_touching_data_in_local_scope_doesnt_leave_it_locked():
    z = np.arange(10.0)
    assert z.flags.writeable is True

    def f(x: np.ndarray):
        for _ in range(4):
            # do a series of operations to ensure
            # "extensive" graphs get garbage collected
            x = mg.ones_like(x) * x
        assert x.data.flags.writeable is False
        return None

    f(z)
    assert z.flags.writeable is True
