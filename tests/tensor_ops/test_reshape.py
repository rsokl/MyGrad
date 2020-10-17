from functools import partial
from numbers import Number

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import reshape
from mygrad.tensor_base import Tensor

from ..custom_strategies import tensors, valid_shapes
from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory


def positional_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, newshape, **kwargs)


def keyword_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, newshape=newshape, **kwargs)


def method_tuple_reshape(arr, newshape, reshaper, **kwargs):
    return arr.reshape(newshape, **kwargs)


def method_unpacked_reshape(arr, newshape, reshaper, **kwargs):
    if newshape == tuple():
        newshape = ((),)
    return (
        arr.reshape(*newshape, **kwargs)
        if isinstance(newshape, tuple)
        else arr.reshape(newshape, **kwargs)
    )


def in_place_reshape(arr, newshape, reshaper, **kwargs):
    to_array = np.asarray if reshaper is np.reshape else Tensor
    arr = +arr  # "touch" array so we can check gradient

    if isinstance(arr, Number):
        arr = to_array(arr)

    arr.shape = newshape
    if isinstance(arr, Tensor) and kwargs.get("constant", False):
        arr._constant = True
    return arr


def test_in_place_reshape_no_autodiff():
    x = mg.arange(10.0)
    with mg.no_autodiff:
        x.shape = (5, 2)
    assert x.shape == (5, 2)
    assert x.creator is None


def test_raising_during_inplace_reshape_doesnt_corrupt_graph():
    x = mg.arange(5.0)
    y = +x
    w = 2 * y
    with pytest.raises(ValueError):
        y.shape = (2, 3)
    w.backward()
    assert_array_equal(w.grad, np.ones_like(w))
    assert_array_equal(y.grad, 2 * np.ones_like(y))
    assert_array_equal(x.grad, 2 * np.ones_like(y))


def test_inplace_reshape_1():
    x = mg.arange(10.0)
    y = x[...]
    x.shape = (2, 5)
    assert x.shape == (2, 5)
    assert y.shape == (10,)
    y.backward()
    assert_array_equal(x.grad, np.ones_like(x))


def test_inplace_reshape_2():
    x0 = mg.arange(10.0)
    x = +x0
    y = x[:6]
    x[-6:] = y
    y.shape = (3, 2)
    y.shape = (1, 6)
    x.shape = (2, 5)

    (2 * y).sum().backward()

    # assert x.shape == (2, 5)
    x_grad = np.zeros(10, dtype=x.dtype)
    x_grad[:6] = 2.0
    assert_array_equal(x.grad, x_grad.reshape(x.shape))
    assert_array_equal(y.grad, 2 * np.ones_like(y))


def test_inplace_reshape_3():
    x0 = mg.arange(10.0)
    x = +x0
    y = x[:5]
    x.shape = (2, 5)
    x[:1] = y[::-1]
    assert_array_equal(x0, mg.arange(10.0))
    assert_array_equal(
        x, np.array([[4.0, 3.0, 2.0, 1.0, 0.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
    )
    assert_array_equal(y, np.array([4.0, 3.0, 2.0, 1.0, 0.0]))

    (x[0] * y).sum().backward()

    assert_array_equal(
        x0.grad, np.array([0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert_array_equal(
        x.grad, np.array([[8.0, 6.0, 4.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    assert_array_equal(y.grad, np.array([4.0, 3.0, 2.0, 1.0, 0.0]))


@given(
    tensor=tensors(), data=st.data(),
)
def test_in_place_reshape(tensor: Tensor, data: st.DataObject):
    assume(tensor.size)

    array = tensor.data.copy()
    newshape = data.draw(valid_shapes(tensor.size, min_len=0), label="newshape")

    tensor.shape = newshape

    array.shape = newshape
    assert_array_equal(array, tensor)

    assert array.base is None
    assert tensor.base is None


@given(
    tensor=tensors(), data=st.data(),
)
def test_in_place_reshape_post_view(tensor: Tensor, data: st.DataObject):
    assume(tensor.size)

    array = tensor.data.copy()
    newshape = data.draw(valid_shapes(tensor.size, min_len=0), label="newshape")

    t1 = tensor[...]
    t1.shape = newshape

    a1 = array[...]
    a1.shape = newshape
    assert_array_equal(array, tensor)
    assert_array_equal(a1, t1)

    assert array.base is None
    assert tensor.base is None
    assert a1.base is array
    assert t1.base is tensor


@pytest.mark.parametrize(
    "reshape_type", [positional_reshape, keyword_reshape],
)
def test_reshape_fwd(reshape_type):
    @fwdprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size)),
    )
    def run_fwd():
        pass

    run_fwd()


@pytest.mark.parametrize(
    "reshape_type", [method_tuple_reshape, method_unpacked_reshape],
)
def test_method_reshape_fwd(reshape_type):
    @fwdprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size)),
        permit_0d_array_as_float=False,
    )
    def run_fwd():
        pass

    run_fwd()


@pytest.mark.parametrize(
    "reshape_type",
    [
        positional_reshape,
        keyword_reshape,
        method_tuple_reshape,
        method_unpacked_reshape,
        in_place_reshape,
    ],
)
def test_reshape_bkwd(reshape_type):
    @backprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size, min_len=0)),
        vary_each_element=True,
    )
    def run_bkwd():
        pass

    run_bkwd()


@pytest.mark.parametrize(
    "bad_input", [tuple(), ((2,), 2), ((2,), 2), (2, (2,)), ((2, (2,)),)]
)
def test_input_validation(bad_input):
    x = Tensor([1, 2])

    with pytest.raises(TypeError):
        x.reshape(*bad_input)


# This test has really weird behavior
# Something is preventing the except-clause
# in Tensor._op to run
# Turn mem-guard off to prevent the leak caused
# by this
@mg.mem_guard_off
def test_input_validation_matches_numpy():
    try:
        np.reshape(np.array(1), *(1, 1))
    except Exception:
        with pytest.raises(Exception):
            reshape(Tensor(1), *(1, 1))
