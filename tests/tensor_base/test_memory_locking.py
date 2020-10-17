from typing import Callable, ContextManager, List, Union

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note
from numpy.testing import assert_allclose

import mygrad as mg
import mygrad._utils.lock_management as mem
from mygrad import (
    Tensor,
    mem_guard_off,
    mem_guard_on,
    turn_memory_guarding_off,
    turn_memory_guarding_on,
)
from tests.custom_strategies import tensors


def writeable(x: Union[np.ndarray, Tensor]) -> bool:
    if isinstance(x, Tensor):
        x = x.data
    return x.flags.writeable


@pytest.mark.parametrize("constant", [True, False])
def test_memory_locks_during_graph(constant: bool):
    x = Tensor([1.0, 2.0], constant=constant)
    assert writeable(x)

    y = +x  # should lock memory
    with pytest.raises(ValueError):
        x.data *= 1
    y.backward()  # should release memory
    x.data *= 1


@pytest.mark.parametrize("constant", [True, False])
def test_unreferenced_op_does_not_lock_data(constant: bool):
    x = mg.arange(10.0, constant=constant)
    y = np.arange(10.0)
    assert writeable(y) is True

    # participating in unreferenced op should not
    # lock y's writeability
    y * x

    assert writeable(y) is True


@pytest.mark.parametrize("func", [lambda x: +x, lambda x: x[...]], ids=["+x", "x[:]"])
@pytest.mark.parametrize("constant", [True, False])
def test_dereferencing_tensor_restores_data_writeability(
    constant: bool, func: Callable[[Tensor], Tensor]
):
    x = mg.arange(2.0, constant=constant)
    data = x.data

    y = +x

    assert writeable(data) is False
    del y
    assert writeable(data) is True, (
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
    assert writeable(y) is False, (
        "`y` is still involved with `z` and `x`; "
        "its writeability should not be restored"
    )
    assert writeable(x) is False, (
        "`x` is still involved with `z` and `y`; "
        "its writeability should not be restored"
    )
    del z
    assert writeable(y) is True
    assert writeable(x) is True


@pytest.mark.parametrize("y_writeable", [True, False])
@given(x=tensors(read_only=st.booleans(), elements=st.floats(-10, 10), shape=(3,)))
def test_view_becomes_writeable_after_base_is_made_writeable(
    x: Tensor, y_writeable: bool
):
    x_was_writeable = writeable(x)

    y = np.arange(float(len(x)))
    y.flags.writeable = y_writeable

    view_y = y[...]
    w = y * x
    z = x * view_y

    del z  # normally would make `view-y` writeable, but `view-y` depends on y
    assert writeable(x) is False
    assert writeable(y) is False
    assert (
        view_y.flags.writeable is False
    ), "view-y can't be made writeable until y is made writeable"

    del w
    assert writeable(x) is x_was_writeable
    assert writeable(y) is y_writeable
    assert writeable(view_y) is y_writeable


def test_touching_data_in_local_scope_doesnt_leave_it_locked():
    z = np.arange(10.0)
    assert z.flags.writeable is True

    def f(x: np.ndarray):
        for _ in range(4):
            # do a series of operations to ensure
            # "extensive" graphs get garbage collected
            x = mg.ones_like(x) * x
        assert writeable(x) is False
        return None

    f(z)
    assert z.flags.writeable is True


@given(
    t1=tensors(shape=(2, 3), fill=st.just(0)), t2=tensors(shape=(4, 4), fill=st.just(0))
)
def test_that_errored_op_doesnt_leave_inputs_locked(t1: Tensor, t2: Tensor):
    t1_was_writeable = t1.data.flags.writeable
    t2_was_writeable = t2.data.flags.writeable
    with pytest.raises(ValueError):
        mg.add(t1, t2)  # shape mismatch
    assert writeable(t1) is t1_was_writeable
    assert writeable(t2) is t2_was_writeable


@pytest.mark.usefixtures("seal_memguard")
def test_suspend_memguard_mid_graph():
    x = mg.arange(3.0)
    arr = np.ones_like(x)
    y = 2 * x
    assert not writeable(x)
    assert not writeable(y)

    with mem_guard_off:
        # should not place locks on arr or z
        # should not place additional lock on y
        z = arr * y

    assert not writeable(y)
    assert writeable(arr)
    assert writeable(z)

    w = arr * z

    assert not writeable(arr)
    assert not writeable(z)
    assert not writeable(w)

    w.backward()
    assert_allclose(x.grad, 2 * np.ones_like(x))

    assert writeable(x)
    assert writeable(y)
    assert writeable(arr)
    assert writeable(z)
    assert writeable(w)


@pytest.mark.usefixtures("seal_memguard")
@pytest.mark.parametrize(
    "memguard_context, expected_value", [(mem_guard_off, False), (mem_guard_on, True)]
)
def test_memguard_context(memguard_context: ContextManager[bool], expected_value: bool):
    mem.MEM_GUARD = not expected_value

    with memguard_context:
        assert mem.MEM_GUARD is expected_value

    assert mem.MEM_GUARD is not expected_value
    mem.MEM_GUARD = True


@pytest.mark.usefixtures("seal_memguard")
@pytest.mark.parametrize(
    "memguard_decorator, expected_value", [(mem_guard_off, False), (mem_guard_on, True)]
)
def test_memguard_decorator(memguard_decorator: Callable, expected_value: bool):
    mem.MEM_GUARD = not expected_value

    @memguard_decorator
    def f():
        assert mem.MEM_GUARD is expected_value

    f()

    assert mem.MEM_GUARD is not expected_value
    mem.MEM_GUARD = True


def assert_nested_contexts_are_consistent():
    initial = mem.MEM_GUARD
    with mem_guard_on:
        with mem_guard_off:
            assert mem.MEM_GUARD is False

            with mem_guard_on:
                assert mem.MEM_GUARD is True

            assert mem.MEM_GUARD is False

    assert mem.MEM_GUARD is initial


@pytest.mark.usefixtures("seal_memguard")
@given(x=tensors(read_only=st.booleans(), shape=(1,)))
def test_inverted_writeability_context(x: Tensor):
    was_writeable = writeable(x)

    turn_memory_guarding_off()
    y = +x

    assert writeable(x) is was_writeable
    assert writeable(y)

    with mem_guard_on:
        z = +x

    assert not writeable(z)
    assert not writeable(x)
    z.backward()

    assert writeable(z)
    assert writeable(x) is was_writeable

    turn_memory_guarding_on()


def compose(iter_of_funcs):
    """(..., f, g, h) -> ... ∘ f ∘ g ∘ h"""
    funcs = list(iter_of_funcs)
    f = funcs.pop()
    for g in funcs[::-1]:
        f = g(f)
    return f


@pytest.mark.usefixtures("seal_memguard")
@given(st.lists(st.sampled_from([mem_guard_on, mem_guard_off]), min_size=0))
def test_nested_contexts(sequence_of_contexts):
    f = (
        compose(sequence_of_contexts)(assert_nested_contexts_are_consistent)
        if sequence_of_contexts
        else assert_nested_contexts_are_consistent
    )
    f()


@pytest.mark.usefixtures("seal_memguard")
def test_documented_mutation():
    turn_memory_guarding_off()
    x = np.arange(3.0)
    y = mg.ones_like(x)
    z = x * y
    x[:] = 0  # mutates x, corrupting state associated with z
    z.backward()
    assert y.grad.sum() == 0.0
    turn_memory_guarding_on()


@pytest.mark.usefixtures("seal_memguard")
def test_documented_guarding():
    import numpy as np

    import mygrad as mg

    # memory guarding is on by default
    x = np.arange(3.0)
    y = mg.ones_like(x)
    z = x * y
    try:
        x[:] = 0  # raises because `x` is made read-only
    except ValueError:
        pass
    z.backward()
    assert_allclose(y.grad, np.array([0.0, 1.0, 2.0]))


def mem_off():
    turn_memory_guarding_off()
    assert mem.MEM_GUARD is False


def mem_on():
    turn_memory_guarding_on()
    assert mem.MEM_GUARD is True


@pytest.mark.usefixtures("seal_memguard")
@given(st.lists(st.sampled_from([mem_on, mem_off])))
def test_turn_memory_guarding_on_off(calls: List[Callable]):
    """
    Call arbitrary sequency off mem-guard on/off;
    Ensure mem_guard_active always matches MEM_GUARD
    """
    assert mg.mem_guard_active() is mem.MEM_GUARD

    for n, call in enumerate(calls):
        note(f"{call}  (call-{n}")
        call()
        assert mg.mem_guard_active() is mem.MEM_GUARD

    turn_memory_guarding_on()
