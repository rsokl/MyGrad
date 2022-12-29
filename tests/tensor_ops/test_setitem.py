from typing import Callable

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad._tensor_core_ops.indexing import (
    _arr,
    _is_bool_array_index,
    _is_int_array_index,
)
from mygrad.tensor_base import Tensor

from ..custom_strategies import (
    adv_integer_index,
    arbitrary_indices,
    basic_indices,
    broadcastable_shapes,
    tensors,
)
from ..utils.numerical_gradient import numerical_gradient_full


# test utilties used by setitem
def test_arr_util():
    assert_array_equal(_arr(2, 2), np.arange(4).reshape(2, 2))
    assert_array_equal(_arr(4, 3), np.arange(12).reshape(4, 3))


@given(tensors(elements=st.floats(-10, 10)))
def test_setitem_mutates_input(x: Tensor):
    assume(x.size)
    data_copy = np.array(x, copy=True)
    data_view = np.asarray(x)
    with mg.no_autodiff:
        x[...] = 1 + x
    assert np.all(x == data_view)
    assert np.all(data_copy != data_view)


@pytest.mark.parametrize(
    ("arr", "truth"),
    [
        ((0, 0), False),
        ((np.array([True]),), False),
        ((np.array([True]), [1]), True),
        ((np.array([True]), [1]), True),
        ((np.array([1]), [1]), True),
        ((np.array([True]), 1), False),
        ((np.array([True]), slice(None)), False),
    ],
)
def test_int_array_test(arr, truth):
    assert _is_int_array_index(arr) is truth


@pytest.mark.parametrize(
    ("arr", "truth"),
    [
        ((0, 0), False),
        ((np.array([True]),), True),
        ((np.array([True]), np.array([False])), False),
        ((np.array([1]), [1]), False),
        ((np.array([True]), 1), False),
        ((np.array([True]), slice(None)), False),
    ],
)
def test_bool_array_test(arr, truth):
    assert _is_bool_array_index(arr) is truth


def setitem(x, y, index):
    x_copy = np.copy(x)
    x_copy[index] = y
    return x_copy


class set_item_test_factory:
    def __init__(
        self,
        array_strat: st.SearchStrategy[np.ndarray],
        index_strat: Callable[[np.ndarray], st.SearchStrategy],
        value_strat: Callable[[np.ndarray], st.SearchStrategy[np.ndarray]],
    ):
        self.array_strat = array_strat
        self.index_strat = index_strat
        self.value_strat = value_strat

    def __call__(self, f: Callable) -> Callable[[], None]:
        """Wraps an empty function to populate it with the test function"""

        @given(x=self.array_strat, data=st.data())
        def wrapper(x: np.ndarray, data: st.DataObject):

            index = data.draw(self.index_strat(x), label="index")

            try:
                o = np.asarray(x[index])
            except IndexError:
                assume(False)
                return

            note(f"x[index]: {o}")
            y = data.draw(self.value_strat(o), label="y")

            numpy_x = np.copy(x)
            numpy_y = np.copy(y)

            mygrad_x = Tensor(np.copy(x))
            mygrad_y = Tensor(np.copy(y))
            mygrad_x1 = +mygrad_x

            try:
                numpy_x[index] = numpy_y  # don't permit invalid set-items
            except Exception:
                assume(False)
                return

            grad = data.draw(
                hnp.arrays(
                    shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True
                ),
                label="grad",
            )

            mygrad_x1[index] = mygrad_y
            out = (mygrad_x1 * grad).sum()  # type: Tensor
            out.backward()

            assert_allclose(
                mygrad_y.data,
                numpy_y,
                err_msg="After `x[index] = y`, tensor-y does not match numpy-y",
            )

            dx, dy = numerical_gradient_full(
                setitem, x, y, back_grad=grad, kwargs=dict(index=index)
            )

            assert_allclose(
                mygrad_x1.grad,
                grad,
                err_msg="After `x[index] = y`, x.grad does not match the expected numerical gradient",
            )
            assert_allclose(
                mygrad_x.grad,
                dx,
                err_msg="After `x[index] = y`, x.grad does not match the expected numerical gradient",
            )
            assert_allclose(
                mygrad_y.grad,
                dy,
                err_msg="After `x[index] = y`, y.grad does not match the expected numerical gradient",
            )

            assert not mygrad_y._ops
            assert not mygrad_x._ops
            assert not mygrad_x1._ops

        return wrapper


def test_setitem_multiple_input():
    """
    Ensures proper backprop through computational graph
    in which variable that is set on serves as multiple
    inputs to a single operation.

    Ensures that clear-graph works properly.
    """
    from mygrad import add_sequence

    x = Tensor([1.0])
    y = x + 0

    assert_array_equal(y.data, np.array([1.0]))

    o = add_sequence(y, y, y)
    y[0] = 4

    assert_array_equal(y.data, np.array([4.0]))

    f = o * y  # 3 * 4
    f.backward()

    assert_array_equal(o.data, np.array([3.0]))
    assert_array_equal(f.data, np.array([12.0]))

    assert_array_equal(x.grad, np.array([12.0]))
    assert_array_equal(o.grad, np.array([4.0]))
    assert_array_equal(y.grad, np.array([3.0]))

    assert not x._ops
    assert not y._ops
    assert not o._ops
    assert not f._ops


@given(x_constant=st.booleans(), y_constant=st.booleans(), data=st.data())
def test_setitem_sanity_check(x_constant, y_constant, data):
    """Ensure proper setitem behavior for all combinations of constant/variable Tensors"""
    x = Tensor([1.0, 2.0, 3.0, 4.0], constant=x_constant)
    w = 4 * x

    as_tensor = data.draw(st.booleans()) if y_constant else True
    y = Tensor([1.0, 0.0], constant=y_constant) if as_tensor else np.array([1.0, 0.0])

    w[::2] = np.array([-1.0, -2.0]) * y
    assert_allclose(np.array((-1.0, 8.0, 0.0, 16.0)), w.data)
    w.sum().backward()

    assert isinstance(w, Tensor)
    assert_allclose(w.data, np.array([-1.0, 8.0, 0.0, 16.0]))

    if not w.constant:
        assert w.grad is not None
        assert_allclose(w.grad, np.ones_like(w.data))
    assert w.constant is x.constant

    if x.constant:
        pass
        assert x.grad is None
    else:
        assert_allclose(x.grad, np.array([0.0, 4.0, 0.0, 4.0]))

    if as_tensor:
        if w.constant or y.constant:
            assert y.grad is None
        else:
            assert_allclose(y.grad, np.array([-1.0, -2.0]))

    assert not w._ops, "clear-graph with clear-graph failed"
    assert not x._ops, "clear-graph with clear-graph failed"

    if as_tensor:
        assert not y._ops, "clear-graph failed"


def test_setitem_downstream_doesnt_affect_upstream_backprop():
    """Test that upstream computational graph is not affected by downstream set-item"""
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = Tensor([-1.0, -2.0, -3.0, -4.0])

    z = x * y
    y[:] = 0

    z.backward()

    assert_allclose(np.ones_like(z.data), z.grad, err_msg=f"{type(z.grad)}")
    assert_allclose(np.array([-1.0, -2.0, -3.0, -4.0]), x.grad)
    assert_allclose(np.array([0.0, 0.0, 0.0, 0.0]), y.data)
    assert y.grad is None


@pytest.mark.parametrize("x_constant", [True, False])
@pytest.mark.parametrize("y_constant", [True, False])
def test_setitem_doesnt_mutate_upstream_nodes(x_constant: bool, y_constant: bool):
    """Ensure setitem doesn't mutate variable non-constant tensor"""
    x = Tensor([1.0, 2.0], constant=x_constant)
    y = Tensor([3.0, 4.0], constant=y_constant)
    z = x + y
    y[:] = 0
    y_old = z.creator.variables[-1]  # version of y that participated in x + y
    assert_allclose(np.array([3.0, 4.0]), y_old)
    assert_allclose(np.array([0.0, 0.0]), y)


@settings(deadline=None, max_examples=1000)
@set_item_test_factory(
    array_strat=hnp.arrays(
        shape=hnp.array_shapes(min_side=0, max_side=4, min_dims=0, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    index_strat=lambda x: basic_indices(x.shape),
    value_strat=lambda o: (
        hnp.arrays(
            # Permit shapes that are broadcast-compatible with x[index]
            # The only excess dimensions permitted in this shape are
            # leading singletons
            shape=broadcastable_shapes(o.shape).map(
                lambda _x: tuple(
                    1 if (len(_x) - n) > o.ndim else s for n, s in enumerate(_x)
                )
            ),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(np.array)
    ),
)
def test_setitem_basic_index():
    pass


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    index_strat=lambda x: adv_integer_index(x.shape),
    value_strat=lambda o: (
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim, max_side=max(o.shape)),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(np.asarray)
    ),
)
def test_setitem_adv_int_index():
    pass


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    index_strat=lambda x: hnp.arrays(shape=x.shape, dtype=bool),
    value_strat=lambda o: (
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim, max_side=max(o.shape)),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(np.asarray)
    ),
)
def test_setitem_adv_bool_index():
    pass


rows = np.array([0, 3], dtype=np.intp)
columns = np.array([0, 2], dtype=np.intp)
index = np.ix_(rows, columns)


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    index_strat=lambda x: st.just(index),
    value_strat=lambda o: hnp.arrays(
        shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
)
def test_setitem_broadcast_index():
    pass


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    index_strat=lambda x: st.just((slice(1, 2), [1, 2])),
    value_strat=lambda o: hnp.arrays(
        shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
)
def test_setitem_mixed_index():
    pass


rows2 = np.array([False, True, False, True])
columns2 = np.array([0, 2], dtype=np.intp)
index2 = (rows2, columns2)


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    index_strat=lambda x: st.just(index2),
    value_strat=lambda o: hnp.arrays(
        shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
)
def test_setitem_broadcast_bool_index():
    """index mixes boolean and int-array indexing"""


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    index_strat=lambda x: st.just(
        (np.array([False, True, False, True]), np.newaxis, slice(None))
    ),
    value_strat=lambda o: hnp.arrays(
        shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
)
def test_setitem_bool_basic_index():
    """index mixes boolean and basic indexing"""


@settings(deadline=None)
@set_item_test_factory(
    array_strat=hnp.arrays(shape=(3, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    index_strat=lambda x: hnp.arrays(shape=(2, 3), dtype=bool).map(
        lambda _x: (_x[0], _x[1])
    ),
    value_strat=lambda o: hnp.arrays(
        shape=broadcastable_shapes(o.shape, max_dims=o.ndim, max_side=max(o.shape)),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    )
    if o.shape and o.size
    else st.floats(-10.0, 10.0).map(np.asarray),
)
def test_setitem_bool_axes_index():
    """index consists of boolean arrays specified for each axis"""


@settings(deadline=None, max_examples=1000)
@set_item_test_factory(
    array_strat=hnp.arrays(
        shape=hnp.array_shapes(min_side=0, max_side=4, min_dims=0, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    index_strat=lambda x: arbitrary_indices(x.shape),
    value_strat=lambda o: (
        hnp.arrays(
            # Permit shapes that are broadcast-compatible with x[index]
            # The only excess dimensions permitted in this shape are
            # leading singletons
            shape=broadcastable_shapes(o.shape).map(
                lambda _x: tuple(
                    1 if (len(_x) - n) > o.ndim else s for n, s in enumerate(_x)
                )
            ),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(np.array)
    ),
)
def test_setitem_arbitrary_index():
    """test arbitrary indices"""
