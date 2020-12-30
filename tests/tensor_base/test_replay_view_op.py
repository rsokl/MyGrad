from typing import Callable

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from numpy.testing import assert_array_equal

from mygrad import Tensor, einsum, reshape
from mygrad.errors import DisconnectedView

from ..custom_strategies import basic_indices, tensors, valid_shapes


def positional_reshape(arr, newshape, **kwargs):
    return reshape(arr, newshape, **kwargs)


def keyword_reshape(arr, newshape, **kwargs):
    return reshape(arr, newshape=newshape, **kwargs)


def method_tuple_reshape(arr, newshape, **kwargs):
    return arr.reshape(newshape, **kwargs)


def method_unpacked_reshape(arr, newshape, **kwargs):
    if newshape == tuple():
        newshape = ((),)
    return (
        arr.reshape(*newshape, **kwargs)
        if isinstance(newshape, tuple)
        else arr.reshape(newshape, **kwargs)
    )


def reshape_args(tensor, data):
    return (
        tuple(),
        dict(
            newshape=data.draw(valid_shapes(tensor.size, min_len=0), label="newshape")
        ),
    )


def getitem(tensor, index, **kwargs):
    return tensor[index]


def basic_index_args(tensor, data):
    index = data.draw(basic_indices(tensor.shape), label="index")
    return (index,), dict()


einsum_strat = tensors(shape=st.integers(1, 5).map(lambda x: (x, x)))


def einsum_view_string(tensor, optimize, constant=False):
    return einsum("ii->i", tensor, optimize=optimize, constant=constant)


def einsum_view_int(tensor, optimize, constant=False):
    return einsum(tensor, [0, 0], [0], optimize=optimize, constant=constant)


def einsum_args(tensor, data):
    return tuple(), dict(optimize=data.draw(st.booleans()))


def no_args(*args, **kwargs):
    return tuple(), dict()


simple_tensors = tensors().filter(lambda t: t.size)


@pytest.mark.parametrize(
    "func, tensor_a_strat, arg_draw",
    [
        (einsum_view_string, einsum_strat, einsum_args),
        (einsum_view_int, einsum_strat, einsum_args),
        (positional_reshape, simple_tensors, reshape_args),
        (keyword_reshape, simple_tensors, reshape_args),
        (method_tuple_reshape, simple_tensors, reshape_args),
        (method_unpacked_reshape, simple_tensors, reshape_args),
        (getitem, simple_tensors, basic_index_args),
    ],
)
@given(
    op_constant=st.booleans(), data=st.data(),
)
def test_view_func_replay_op_mirrors_standard_op(
    func: Callable,
    tensor_a_strat: st.SearchStrategy,
    arg_draw: Callable,
    op_constant: bool,
    data: st.DataObject,
):

    tensor_a = data.draw(tensor_a_strat, label="tensor_a")
    args, kwargs = arg_draw(tensor=tensor_a, data=data)

    a_view = func(tensor_a, *args, **kwargs, constant=op_constant)

    assume(a_view.base is not None)

    tensor_b = data.draw(tensors(shape=tensor_a.shape), label="tensor_b")

    b_view = func(tensor_b, *args, **kwargs, constant=op_constant)

    # should behave identically to `func(tensor_b, ...)`
    b_via_replay = a_view._replay_op(tensor_b)

    assert_array_equal(b_via_replay, b_view)
    assert isinstance(b_via_replay.creator, type(b_view.creator))
    assert b_via_replay.constant is b_view.constant
    assert b_via_replay.base is b_view.base


@given(tensors())
def test_op_replay_without_creator_raises(x: Tensor):
    with pytest.raises(DisconnectedView):
        x._replay_op(x)
