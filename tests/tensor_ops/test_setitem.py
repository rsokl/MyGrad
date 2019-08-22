import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from numpy.testing import assert_allclose, assert_array_equal

from mygrad.tensor_base import Tensor
from mygrad.tensor_core_ops.indexing import (
    _arr,
    _is_bool_array_index,
    _is_int_array_index,
)

from ..custom_strategies import adv_integer_index, basic_index, broadcastable_shapes
from ..utils.numerical_gradient import numerical_gradient_full


# test utilties used by setitem
def test_arr_util():
    assert_array_equal(_arr(2, 2), np.arange(4).reshape(2, 2))
    assert_array_equal(_arr(4, 3), np.arange(12).reshape(4, 3))


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


def test_setitem_multiple_input():
    """
    Ensures proper backprop through computational graph
    in which variable that is set on serves as multiple
    inputs to a single operation.

    Ensures that null-gradient and clear-graph works properly.
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

    f.null_gradients()
    assert x.grad is None and not x._ops and not x._accum_ops
    assert y.grad is None and not y._ops and not y._accum_ops
    assert o.grad is None and not o._ops and not o._accum_ops
    assert f.grad is None and not f._ops and not f._accum_ops


@given(x_constant=st.booleans(), y_constant=st.booleans(), data=st.data())
def test_setitem_sanity_check(x_constant, y_constant, data):
    """ Ensure proper setitem behavior for all combinations of constant/variable Tensors"""
    x = Tensor([1.0, 2.0, 3.0, 4.0], constant=x_constant)
    w = 4 * x

    as_tensor = data.draw(st.booleans()) if y_constant else True
    y = Tensor([1.0, 0.0], constant=y_constant) if as_tensor else np.array([1.0, 0.0])

    w[::2] = np.array([-1.0, -2.0]) * y
    assert_allclose(np.array((-1.0, 8.0, 0.0, 16.0)), w.data)
    w.sum().backward()

    assert isinstance(w, Tensor)
    assert_allclose(w.data, np.array([-1.0, 8.0, 0.0, 16.0]))
    assert w.constant is (x.constant and (not as_tensor or y.constant))

    if x.constant:
        assert x.grad is None
    else:
        assert_allclose(x.grad, np.array([0.0, 4.0, 0.0, 4.0]))

    if as_tensor:
        if y.constant:
            assert y.grad is None
        else:
            assert_allclose(y.grad, np.array([-1.0, -2.0]))

    w.null_gradients()
    assert x.grad is None, "null_gradients failed"

    if as_tensor:
        assert y.grad is None, "null_gradients failed"


def test_setitem_sanity_check2():
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    y = Tensor([-1.0, -2.0, -3.0, -4.0])

    z = x * y
    y[:] = 0

    z.backward()
    assert_allclose(np.array([-1.0, -2.0, -3.0, -4.0]), x.grad)
    assert_allclose(np.array([0.0, 0.0, 0.0, 0.0]), y.data)
    assert y.grad is None


def test_no_mutate():
    """ Ensure setitem doesn't mutate variable non-constant tensor"""
    x = Tensor([1.0, 2.0])
    y = Tensor([3.0, 4.0])
    x + y
    y[:] = 0
    y_old = x._ops.pop().variables[-1]  # version of y that participated in x + y
    assert_allclose(np.array([3.0, 4.0]), y_old.data)
    assert_allclose(np.array([0.0, 0.0]), y.data)


@settings(deadline=None, max_examples=1000)
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    data=st.data(),
)
def test_setitem_basic_index(x: np.ndarray, data: st.DataObject):
    """ index conforms strictly to basic indexing """
    index = data.draw(basic_index(x.shape), label="index")
    o = np.asarray(x[index])

    note("x[index]: {}".format(o))
    y = data.draw(
        (
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
            else st.floats(-10.0, 10.0).map(lambda _x: np.array(_x))
        ),
        label="y",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]

    try:
        x0[index] = y0  # don't permit invalid set-items
    except ValueError:
        assume(False)
        return

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    data=st.data(),
)
def test_setitem_adv_int_index(x, data):
    """ index consists of a tuple of integer-valued arrays """
    index = data.draw(adv_integer_index(x.shape), label="index")
    o = np.asarray(x[index])
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim, max_side=max(o.shape)),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(lambda _x: np.array(_x)),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))

    try:
        x0[index] = y0  # don't permit invalid set-items
    except ValueError:
        assume(False)
        return

    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    data=st.data(),
)
def test_setitem_adv_bool_index(x, data):
    """ index consists of a single boolean-valued array """
    index = data.draw(hnp.arrays(shape=x.shape, dtype=bool), label="index")
    o = np.asarray(x[index])
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim, max_side=max(o.shape)),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(lambda _x: np.array(_x)),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    try:
        x0[index] = y0  # don't permit invalid set-items
    except ValueError:
        assume(False)
        return

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    data=st.data(),
)
def test_setitem_broadcast_index(x, data):
    """ index is two broadcast-compatible integer arrays"""
    # test broadcast-compatible int-arrays
    rows = np.array([0, 3], dtype=np.intp)
    columns = np.array([0, 2], dtype=np.intp)
    index = np.ix_(rows, columns)
    o = np.asarray(x[index])
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        ),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    x0[index] = y0
    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    data=st.data(),
)
def test_setitem_mixed_index(x, data):
    """ index is mixes basic and advanced int-array indexing"""
    index = (slice(1, 2), [1, 2])
    o = np.asarray(x[index])
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        ),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    try:
        x0[index] = y0  # don't permit invalid set-items
    except ValueError:
        assume(False)
        return

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    x0[index] = y0
    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    data=st.data(),
)
def test_setitem_broadcast_bool_index(x, data):
    """ index mixes boolean and int-array indexing"""
    rows = np.array([False, True, False, True])
    columns = np.array([0, 2], dtype=np.intp)
    index = np.ix_(rows, columns)
    o = np.asarray(x[index])
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        ),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    x0[index] = y0
    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(shape=(4, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    data=st.data(),
)
def test_setitem_bool_basic_index(x, data):
    """ index mixes boolean and basic indexing"""
    index = (np.array([False, True, False, True]), np.newaxis, slice(None))
    o = np.asarray(x[index])
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        ),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    try:
        x0[index] = y0  # don't permit invalid set-items
    except ValueError:
        assume(False)
        return

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@settings(deadline=None)
@given(
    x=hnp.arrays(shape=(3, 3), dtype=float, elements=st.floats(-10.0, 10.0)),
    data=st.data(),
)
def test_setitem_bool_axes_index(x, data):
    """ index consists of boolean arrays specified for each axis """
    index = data.draw(
        st.tuples(
            hnp.arrays(shape=(3,), dtype=bool), hnp.arrays(shape=(3,), dtype=bool)
        )
    )
    try:
        o = np.asarray(x[index])
    except IndexError:
        return None
    y = data.draw(
        hnp.arrays(
            shape=broadcastable_shapes(o.shape, max_dims=o.ndim, max_side=max(o.shape)),
            dtype=float,
            elements=st.floats(-10.0, 10.0),
        )
        if o.shape and o.size
        else st.floats(-10.0, 10.0).map(lambda _x: np.array(_x)),
        label="y",
    )

    grad = data.draw(
        hnp.arrays(shape=x.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    x0 = np.copy(x)
    y0 = np.copy(y)

    try:
        x0[index] = y0  # don't permit invalid set-items
    except ValueError:
        assume(False)
        return

    x_arr = Tensor(np.copy(x))
    y_arr = Tensor(np.copy(y))
    x1_arr = x_arr[:]
    x1_arr[index] = y_arr
    (x1_arr * grad).sum().backward()

    assert_allclose(x1_arr.data, x0)
    assert_allclose(y_arr.data, y0)

    dx, dy = numerical_gradient_full(
        setitem, x, y, back_grad=grad, kwargs=dict(index=index)
    )

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)
