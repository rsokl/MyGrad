from mygrad.tensor_base import Tensor
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from ..utils.numerical_gradient import numerical_gradient_full
from ..custom_strategies import basic_index, adv_integer_index, broadcastable_shape

from mygrad.tensor_core_ops.indexing import _arr, _is_bool_array_index, _is_int_array_index


# test utilties used by setitem
def test_arr_util():
    assert_array_equal(_arr(2, 2), np.arange(4).reshape(2, 2))
    assert_array_equal(_arr(4, 3), np.arange(12).reshape(4, 3))


def test_int_array_test():
    assert _is_int_array_index((0, 0)) is False
    assert _is_int_array_index((np.array([True]), )) is False
    assert _is_int_array_index((np.array([True]), [1])) is True
    assert _is_int_array_index((np.array([1]), [1])) is True
    assert _is_int_array_index((np.array([True]), 1)) is False
    assert _is_int_array_index((np.array([True]), slice(None))) is False


def test_bool_array_test():
    assert _is_bool_array_index((0, 0)) is False
    assert _is_bool_array_index((np.array([True]),)) is True
    assert _is_bool_array_index((np.array([True]), np.array([False]))) is False
    assert _is_bool_array_index((np.array([1]), [1])) is False
    assert _is_bool_array_index((np.array([True]), 1)) is False
    assert _is_bool_array_index((np.array([True]), slice(None))) is False


def setitem(x, y, index):
    x_copy = np.copy(x)
    x_copy[index] = y
    return x_copy


@given(x_constant=st.booleans(),
       y_constant=st.booleans(),
       data=st.data())
def test_setitem_sanity_check(x_constant, y_constant, data):
    """ Ensure proper setitem behavior for all combinations of constant/variable Tensors"""
    x = Tensor([1., 2., 3., 4.], constant=x_constant)
    w = 4 * x

    as_tensor = data.draw(st.booleans()) if y_constant else True
    y = Tensor([1., 0.], constant=y_constant) if as_tensor else np.array([1., 0.])

    w[::2] = np.array([-1., -2.]) * y
    assert_allclose(np.array((-1., 8., 0., 16.)), w.data)
    w.sum().backward()

    assert isinstance(w, Tensor)
    assert_allclose(w.data, np.array([-1., 8., 0., 16.]))
    assert w.constant is (x.constant and (not as_tensor or y.constant))

    if x.constant:
        assert x.grad is None
    else:
        assert_allclose(x.grad, np.array([0., 4., 0., 4.]))

    if as_tensor:
        if y.constant:
            assert y.grad is None
        else:
            assert_allclose(y.grad, np.array([-1., -2.]))

    w.null_gradients()
    assert x.grad is None, "null_gradients failed"

    if as_tensor:
        assert y.grad is None, "null_gradients failed"


def test_setitem_sanity_check2():
    x = Tensor([1., 2., 3., 4.])
    y = Tensor([-1., -2., -3., -4.])

    z = x * y
    y[:] = 0

    z.backward()
    assert_allclose(np.array([-1., -2., -3., -4.]), x.grad)
    assert_allclose(np.array([0., 0., 0., 0.]), y.data)
    assert y.grad is None


def test_no_mutate():
    """ Ensure setitem doesn't mutate variable non-constant tensor"""
    x = Tensor([1., 2.])
    y = Tensor([3., 4.])
    x + y
    y[:] = 0
    y_old = x._ops[0].variables[-1]  # version of y that participated in x + y
    assert_allclose(np.array([3., 4.]), y_old.data)
    assert_allclose(np.array([0., 0.]), y.data)


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_basic_index(x, data):
    """ index conforms strictly to basic indexing """
    index = data.draw(basic_index(x.shape), label="index")
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_adv_int_index(x, data):
    """ index consists of a tuple of integer-valued arrays """
    index = data.draw(adv_integer_index(x.shape), label="index")
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_adv_bool_index(x, data):
    """ index consists of a single boolean-valued array """
    index = data.draw(hnp.arrays(shape=x.shape, dtype=bool), label="index")
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=(4, 3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_broadcast_index(x, data):
    """ index is two broadcast-compatible integer arrays"""
    # test broadcast-compatible int-arrays
    rows = np.array([0, 3], dtype=np.intp)
    columns = np.array([0, 2], dtype=np.intp)
    index = np.ix_(rows, columns)
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=(4, 3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_mixed_index(x, data):
    """ index is mixes basic and advanced int-array indexing"""
    index = (slice(1, 2), [1, 2])
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=(4, 3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_broadcast_bool_index(x, data):
    """ index mixes boolean and int-array indexing"""
    rows = np.array([False,  True, False,  True])
    columns = np.array([0, 2], dtype=np.intp)
    index = np.ix_(rows, columns)
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=(4, 3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_bool_basic_index(x, data):
    """ index mixes boolean and basic indexing"""
    index = (np.array([False,  True, False,  True]), np.newaxis, slice(None))
    o = np.asarray(x[index])
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)


@given(x=hnp.arrays(shape=(3, 3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_setitem_bool_axes_index(x, data):
    """ index consists of boolean arrays specified for each axis """
    index = data.draw(st.tuples(hnp.arrays(shape=(3,), dtype=bool), hnp.arrays(shape=(3,), dtype=bool)))
    try:
        o = np.asarray(x[index])
    except IndexError:
        return None
    y = data.draw(hnp.arrays(shape=broadcastable_shape(o.shape, max_dim=o.ndim),
                             dtype=float,
                             elements=st.floats(-10., 10.)), label="y")

    grad = data.draw(hnp.arrays(shape=x.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

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

    dx, dy = numerical_gradient_full(setitem, x, y, back_grad=grad,
                                     as_decimal=True, kwargs=dict(index=index))

    assert_allclose(x_arr.grad, dx)
    assert_allclose(y_arr.grad, dy)
