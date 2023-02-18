from itertools import permutations
from typing import List

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, infer, settings
from numpy.testing import assert_allclose
from pytest import raises

from mygrad import (
    Tensor,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_to,
    expand_dims,
    moveaxis,
    no_autodiff,
    ravel,
    repeat,
    roll,
    swapaxes,
    transpose,
)
from mygrad.typing import ArrayLike
from tests.utils.functools import add_constant_passthrough
from tests.utils.wrappers import adds_constant_arg

from .custom_strategies import valid_axes
from .utils.numerical_gradient import numerical_gradient_full
from .wrappers.uber import backprop_test_factory, fwdprop_test_factory


def test_input_validation():
    x = Tensor([[1, 2]])

    with raises(TypeError):
        transpose(x, (0,), 1)

    with raises(TypeError):
        x.transpose((0,), 1)


@settings(deadline=None)
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, max_dims=5),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    data=st.data(),
)
def test_transpose(x, data):
    axes = data.draw(
        valid_axes(x.ndim, min_dim=x.ndim, max_dim=x.ndim).map(
            lambda out: (out,) if isinstance(out, int) else out
        ),
        label="axes",
    )

    x_arr = Tensor(np.copy(x))

    o = transpose(x_arr, axes, constant=False)
    grad = data.draw(
        hnp.arrays(shape=o.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )

    o.backward(grad)

    def f(x):
        return np.transpose(x, axes)

    assert_allclose(o.data, f(x))

    (dx,) = numerical_gradient_full(f, x, back_grad=grad)

    assert_allclose(x_arr.grad, dx)

    out = transpose(x, constant=True)
    assert out.constant and not x_arr.constant


def test_transpose_property():
    dat = np.arange(6.0).reshape(2, 3)
    x = Tensor(dat)
    f = x.T
    f.backward(dat.T)

    assert_allclose(f.data, dat.T)
    assert_allclose(x.grad, dat)


def test_transpose_method():
    dat = np.arange(24.0).reshape(2, 3, 4)

    for axes in permutations(range(3)):
        # passing tuple of integers
        x = Tensor(dat)
        f = x.transpose(axes)
        f.backward(dat.transpose(axes))

        assert_allclose(f.data, dat.transpose(axes))
        assert_allclose(x.grad, dat)

        # passing integers directly
        x = Tensor(dat)
        f = x.transpose(*axes)
        f.backward(dat.transpose(axes))

        assert_allclose(f.data, dat.transpose(axes), err_msg="{}".format(axes))
        assert_allclose(x.grad, dat, err_msg="{}".format(axes))

    # passing integers directly
    x = Tensor(dat)
    f = x.transpose()
    f.backward(dat.transpose())

    assert_allclose(f.data, dat.transpose())
    assert_allclose(x.grad, dat)

    # check that constant=True works
    x = Tensor(dat)
    f = x.transpose(constant=True)
    assert f.constant and not x.constant

    f = x.transpose(1, 0, 2, constant=True)
    assert f.constant and not x.constant


@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(max_side=2, max_dims=3),
        dtype=float,
        elements=st.floats(-10.0, 10.0),
    ),
    data=st.data(),
)
def test_squeeze(x, data):
    axes = data.draw(valid_axes(x.ndim), label="axes")
    x_arr = Tensor(np.copy(x))
    x_arr2 = Tensor(np.copy(x))

    def f(x):
        return np.squeeze(x, axes)

    try:
        numpy_out = np.squeeze(x, axes)
    except ValueError:
        with raises(ValueError):
            np.squeeze(x_arr, axes)
        return

    o = np.squeeze(x_arr, axes)  # exercises __array_function__
    o_method = x_arr2.squeeze(axes)
    assert_allclose(o.data, numpy_out)
    assert_allclose(o_method.data, numpy_out)

    grad = data.draw(
        hnp.arrays(shape=o.shape, dtype=float, elements=st.floats(1, 10), unique=True),
        label="grad",
    )
    o.backward(grad)
    o_method.backward(grad)

    (dx,) = numerical_gradient_full(f, x, back_grad=grad)

    assert_allclose(x_arr.grad, dx)
    assert_allclose(x_arr2.grad, dx)


def _expand_dims_axis(arr):
    return st.integers(-arr.ndim - 1, arr.ndim)


def _swap_axes_axis(arr):
    return st.integers(-arr.ndim, arr.ndim - 1) if arr.ndim else st.just(0)


def _valid_moveaxis_args(*arrs, **kwargs):
    return len(kwargs["source"]) == len(kwargs["destination"])


def _transpose(x, axes, constant=False):
    return transpose(x, axes, constant=constant)


def _np_transpose(x, axes):
    return np.transpose(x, axes)


@adds_constant_arg
def _transpose_property(x):
    if not isinstance(x, Tensor):
        x = np.asarray(x)
    return x.T


@fwdprop_test_factory(
    mygrad_func=_transpose_property,
    true_func=_transpose_property,
    num_arrays=1,
    permit_0d_array_as_float=False,
)
def test_transpose_property_fwd():
    pass


@backprop_test_factory(
    mygrad_func=_transpose_property,
    true_func=_transpose_property,
    num_arrays=1,
    vary_each_element=True,
)
def test_transpose_property_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=_transpose,
    true_func=_np_transpose,
    num_arrays=1,
    kwargs=dict(axes=lambda x: valid_axes(x.ndim, min_dim=x.ndim, max_dim=x.ndim)),
)
def test_transpose_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(_np_transpose),  # exercises __array_function__
    true_func=_np_transpose,
    num_arrays=1,
    kwargs=dict(axes=lambda x: valid_axes(x.ndim, min_dim=x.ndim, max_dim=x.ndim)),
    vary_each_element=True,
)
def test_transpose_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=expand_dims,
    true_func=np.expand_dims,
    num_arrays=1,
    kwargs=dict(axis=_expand_dims_axis),
)
def test_expand_dims_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(
        np.expand_dims
    ),  # exercises __array_function__
    true_func=np.expand_dims,
    num_arrays=1,
    kwargs=dict(axis=_expand_dims_axis),
    vary_each_element=True,
)
def test_expand_dims_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=moveaxis,
    true_func=np.moveaxis,
    num_arrays=1,
    kwargs=dict(
        source=lambda x: valid_axes(x.ndim, permit_none=False, permit_int=False),
        destination=lambda x: valid_axes(x.ndim, permit_none=False, permit_int=False),
    ),
    assumptions=_valid_moveaxis_args,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=4, max_dims=5)},
)
def test_moveaxis_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.moveaxis),  # exercises __array_function__
    true_func=np.moveaxis,
    num_arrays=1,
    kwargs=dict(
        source=lambda x: valid_axes(x.ndim, permit_none=False, permit_int=False),
        destination=lambda x: valid_axes(x.ndim, permit_none=False, permit_int=False),
    ),
    assumptions=_valid_moveaxis_args,
    vary_each_element=True,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=4, max_dims=5)},
)
def test_moveaxis_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=swapaxes,
    true_func=np.swapaxes,
    num_arrays=1,
    kwargs=dict(axis1=_swap_axes_axis, axis2=_swap_axes_axis),
    index_to_arr_shapes={0: hnp.array_shapes(max_side=3, min_dims=1, max_dims=3)},
)
def test_swapaxes_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.swapaxes),  # exercises __array_function__
    true_func=np.swapaxes,
    num_arrays=1,
    kwargs=dict(axis1=_swap_axes_axis, axis2=_swap_axes_axis),
    vary_each_element=True,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=3, min_dims=1, max_dims=3)},
)
def test_swapaxes_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=Tensor.flatten,
    true_func=np.ndarray.flatten,
    num_arrays=1,
    permit_0d_array_as_float=False,
)
def test_flatten_fwd():
    pass


@backprop_test_factory(
    mygrad_func=Tensor.flatten,
    true_func=np.ndarray.flatten,
    num_arrays=1,
    vary_each_element=True,
)
def test_flatten_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=ravel, true_func=np.ravel, num_arrays=1)
def test_ravel_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.ravel),  # exercises __array_function__
    true_func=np.ravel,
    num_arrays=1,
    vary_each_element=True,
)
def test_ravel_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=broadcast_to,
    true_func=np.broadcast_to,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=0)},
    kwargs=dict(
        shape=lambda arr: hnp.array_shapes(min_dims=0).map(lambda x: x + arr.shape)
    ),
)
def test_broadcast_to_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(
        np.broadcast_to
    ),  # exercise __array_function__
    true_func=np.broadcast_to,
    num_arrays=1,
    vary_each_element=True,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=0)},
    kwargs=dict(
        shape=lambda arr: hnp.array_shapes(min_dims=0).map(lambda x: x + arr.shape)
    ),
)
def test_broadcast_to_bkwd():
    pass


@st.composite
def gen_roll_args(draw, arr):
    shift = draw(st.integers() | st.tuples(*(st.integers() for i in arr.shape)))

    if arr.ndim:
        ax_strat = hnp.valid_tuple_axes(
            arr.ndim,
            **(
                dict(min_size=len(shift), max_size=len(shift))
                if isinstance(shift, tuple)
                else {}
            )
        )
        axis = draw(st.none() | st.integers(-arr.ndim, arr.ndim - 1) | ax_strat)
    else:
        axis = None
    return dict(shift=shift, axis=axis)


@fwdprop_test_factory(
    mygrad_func=roll, true_func=np.roll, num_arrays=1, kwargs=gen_roll_args
)
def test_roll_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.roll),  # exercises __array_function__
    true_func=np.roll,
    num_arrays=1,
    kwargs=gen_roll_args,
    vary_each_element=True,
)
def test_roll_bkwd():
    pass


def gen_int_repeat_args(arr: Tensor) -> st.SearchStrategy[dict]:
    valid_axis = st.none()
    valid_axis |= st.integers(-arr.ndim, arr.ndim - 1) if arr.ndim else st.just(0)
    return st.fixed_dictionaries(
        dict(
            repeats=st.integers(min_value=0, max_value=5),
            axis=valid_axis,
        )
    )


@st.composite
def gen_tuple_repeat_args(draw: st.DataObject.draw, arr: Tensor):
    valid_axis = draw(
        st.none() | (st.integers(-arr.ndim, arr.ndim - 1) if arr.ndim else st.just(0))
    )

    num_repeats = (
        arr.shape[valid_axis] if valid_axis is not None and arr.ndim else arr.size
    )
    repeats = draw(st.tuples(*[st.integers(0, 5)] * num_repeats))
    return dict(
        repeats=repeats,
        axis=valid_axis,
    )


@fwdprop_test_factory(
    mygrad_func=repeat,
    true_func=np.repeat,
    num_arrays=1,
    kwargs=gen_int_repeat_args,
)
def test_repeat_int_repeats_only_fwd():
    pass


@backprop_test_factory(
    mygrad_func=repeat,
    true_func=np.repeat,
    num_arrays=1,
    kwargs=gen_int_repeat_args,
    vary_each_element=True,
)
def test_repeat_int_repeats_only_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=repeat,
    true_func=np.repeat,
    num_arrays=1,
    kwargs=gen_tuple_repeat_args,
)
def test_repeat_tuple_repeats_only_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.repeat),  # exercises __array_function__
    true_func=np.repeat,
    num_arrays=1,
    kwargs=gen_tuple_repeat_args,
    vary_each_element=True,
)
def test_repeat_tuple_repeats_only_bkwd():
    pass


def _wrap_list(x):
    return x if isinstance(x, list) else [x]


@pytest.mark.parametrize("func", [atleast_1d, atleast_2d, atleast_3d])
@given(x=infer, constant=st.none() | st.booleans())
def test_atleast_kd_fixed_point(func, x: List[ArrayLike], constant):
    with no_autodiff:
        out1 = _wrap_list(func(*x, constant=constant))
        out2 = _wrap_list(func(*out1, constant=constant))
        assert len(out1) == len(out2)
        assert all(x is y for x, y in zip(out1, out2))


@fwdprop_test_factory(
    mygrad_func=atleast_1d,
    true_func=np.atleast_1d,
    num_arrays=1,
)
def test_atleast_1d_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.atleast_1d),  # exercises __array_function__
    true_func=np.atleast_1d,
    num_arrays=1,
    vary_each_element=True,
)
def test_atleast_1d_only_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=atleast_2d,
    true_func=np.atleast_2d,
    num_arrays=1,
)
def test_atleast_2d_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.atleast_2d),  # exercises __array_function__
    true_func=np.atleast_2d,
    num_arrays=1,
    vary_each_element=True,
)
def test_atleast_2d_only_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=atleast_3d,
    true_func=np.atleast_3d,
    num_arrays=1,
)
def test_atleast_3d_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.atleast_3d),  # exercises __array_function__
    true_func=np.atleast_3d,
    num_arrays=1,
    vary_each_element=True,
)
def test_atleast_3d_only_bkwd():
    pass
