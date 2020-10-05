from typing import Callable

import numpy as np
import pytest

import mygrad as mg
from mygrad import Tensor


@pytest.mark.parametrize("constant", [True, False])
def test_memory_locks_during_graph(constant: bool):
    x = Tensor([1.0, 2.0], constant=constant)
    assert x.data.flags.writeable

    y = +x  # should lock memory
    with pytest.raises(ValueError):
        x.data *= 1
    y.backward()  # should release memory
    x.data *= 1


@pytest.mark.parametrize("constant", [True, False])
def test_unreferenced_op_does_not_lock_data(constant: bool):
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
    assert data.flags.writeable is True, (
        "x is no longer participating in a graph; its memory should be" "writeable"
    )


@pytest.mark.parametrize("constant", [True, False])
def test_only_final_dereference_restores_writeability(constant: bool):
    x = mg.arange(10.0, constant=constant)
    y = np.arange(10.0)
    assert y.flags.writeable is True

    w = y[...] * x
    z = y * x[...]

    del w
    assert y.flags.writeable is False, (
        "`y` is still involved with `z` and `x`; "
        "its writeability should not be restored"
    )
    assert x.data.flags.writeable is False, (
        "`x` is still involved with `z` and `y`; "
        "its writeability should not be restored"
    )
    del z
    assert y.flags.writeable is True
    assert x.data.flags.writeable is True


@pytest.mark.parametrize("constant", [True, False])
def test_view_becomes_writeable_after_base_is_made_writeable(constant: bool):
    x = mg.arange(10.0, constant=constant)
    y = np.arange(10.0)
    assert y.flags.writeable is True

    view_y = y[...]
    w = y * x
    z = x * view_y

    del z  # normally would make `view-y` writeable, but `view-y` depends on y
    assert x.data.flags.writeable is False
    assert y.flags.writeable is False
    assert (
        view_y.flags.writeable is False
    ), "view-y can't be made writeable until y is made writeable"

    del w
    assert x.data.flags.writeable is True
    assert y.flags.writeable is True
    assert view_y.flags.writeable is True


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
