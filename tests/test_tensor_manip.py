from itertools import permutations
from mygrad.tensor_base import Tensor
from mygrad import transpose, moveaxis, swapaxes, squeeze, expand_dims, ravel, broadcast_to
from numpy.testing import assert_allclose
import numpy as np

from .custom_strategies import valid_axes, broadcastable_shape
from .utils.numerical_gradient import numerical_gradient_full
from .wrappers.uber import fwdprop_test_factory, backprop_test_factory

import hypothesis.extra.numpy as hnp
from hypothesis import given, assume, settings
import hypothesis.strategies as st

from pytest import raises
    

@settings(deadline=None)
@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=4, max_dims=5),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_transpose(x, data):
    axes = data.draw(valid_axes(x.ndim), label="axes")
    if axes is not None:
        assume(len(axes) == x.ndim)

    x_arr = Tensor(np.copy(x))

    o = transpose(x_arr, axes, constant=False)
    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")

    o.backward(grad)

    def f(x): return np.transpose(x, axes)

    assert_allclose(o.data, f(x))

    dx, = numerical_gradient_full(f, x, back_grad=grad)

    assert_allclose(x_arr.grad, dx)

    out = transpose(x, constant=True)
    assert out.constant and not x_arr.constant


def test_transpose_property():
    dat = np.arange(6).reshape(2, 3)
    x = Tensor(dat)
    f = x.T
    f.backward(dat.T)

    assert_allclose(f.data, dat.T)
    assert_allclose(x.grad, dat)


def test_transpose_method():
    dat = np.arange(24).reshape(2, 3, 4)

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


@given(x=hnp.arrays(shape=hnp.array_shapes(max_side=2, max_dims=3),
                    dtype=float,
                    elements=st.floats(-10., 10.)),
       data=st.data())
def test_squeeze(x, data):
    axes = data.draw(valid_axes(x.ndim), label="axes")
    x_arr = Tensor(np.copy(x))
    x_arr2 = Tensor(np.copy(x))

    def f(x): return np.squeeze(x, axes)

    try:
        numpy_out = np.squeeze(x, axes)
    except ValueError as e:
        with raises(ValueError):
            squeeze(x_arr, axes, constant=False)
        return

    o = squeeze(x_arr, axes, constant=False)
    o_method = x_arr2.squeeze(axes)
    assert_allclose(o.data, numpy_out)
    assert_allclose(o_method.data, numpy_out)

    grad = data.draw(hnp.arrays(shape=o.shape,
                                dtype=float,
                                elements=st.floats(1, 10),
                                unique=True),
                     label="grad")
    o.backward(grad)
    o_method.backward(grad)

    dx, = numerical_gradient_full(f, x, back_grad=grad)

    assert_allclose(x_arr.grad, dx)
    assert_allclose(x_arr2.grad, dx)


def _expand_dims_axis(arr):
    return st.integers(-arr.ndim - 1, arr.ndim)


def _swap_axes_axis(arr):
    return st.integers(-arr.ndim, arr.ndim - 1)


def _valid_moveaxis_args(*arrs, **kwargs):
    return len(kwargs["source"]) == len(kwargs["destination"])


@fwdprop_test_factory(mygrad_func=expand_dims, true_func=np.expand_dims,
                      num_arrays=1, kwargs=dict(axis=_expand_dims_axis))
def test_expand_dims_fwd():
    pass


@backprop_test_factory(mygrad_func=expand_dims, true_func=np.expand_dims,
                       num_arrays=1, kwargs=dict(axis=_expand_dims_axis),
                       vary_each_element=True)
def test_expand_dims_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=moveaxis, true_func=np.moveaxis,
                      num_arrays=1, kwargs=dict(source=lambda x: valid_axes(x.ndim, permit_none=False),
                                                destination=lambda x: valid_axes(x.ndim, permit_none=False)),
                      assumptions=_valid_moveaxis_args,
                      index_to_arr_shapes={0: hnp.array_shapes(max_side=4, max_dims=5)})
def test_moveaxis_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=moveaxis, true_func=np.moveaxis,
                       num_arrays=1, kwargs=dict(source=lambda x: valid_axes(x.ndim, permit_none=False),
                                                 destination=lambda x: valid_axes(x.ndim, permit_none=False)),
                       assumptions=_valid_moveaxis_args,
                       vary_each_element=True,
                       index_to_arr_shapes={0: hnp.array_shapes(max_side=4, max_dims=5)})
def test_moveaxis_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=swapaxes, true_func=np.swapaxes,
                      num_arrays=1, kwargs=dict(axis1=_swap_axes_axis, axis2=_swap_axes_axis))
def test_swapaxes_fwd():
    pass


@backprop_test_factory(mygrad_func=swapaxes, true_func=np.swapaxes,
                       num_arrays=1, kwargs=dict(axis1=_swap_axes_axis, axis2=_swap_axes_axis),
                       vary_each_element=True)
def test_swapaxes_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=Tensor.flatten, true_func=np.ndarray.flatten,
                      num_arrays=1)
def test_flatten_fwd():
    pass


@backprop_test_factory(mygrad_func=Tensor.flatten, true_func=np.ndarray.flatten,
                       num_arrays=1, vary_each_element=True)
def test_flatten_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=ravel, true_func=np.ravel,
                      num_arrays=1)
def test_ravel_fwd():
    pass


@backprop_test_factory(mygrad_func=ravel, true_func=np.ravel,
                       num_arrays=1, vary_each_element=True)
def test_ravel_bkwd():
    pass


def _is_broadcastable(*arrs, **kwargs):
    x = arrs[0]
    return x.shape == kwargs["shape"][-x.ndim:]


@fwdprop_test_factory(mygrad_func=broadcast_to, true_func=np.broadcast_to,
                      num_arrays=1,
                      index_to_arr_shapes={0: hnp.array_shapes(min_dims=0)},
                      kwargs=dict(shape=lambda arr: broadcastable_shape(arr.shape, min_dim=arr.ndim)),
                      assumptions=_is_broadcastable)
def test_broadcast_to_fwd():
    pass


@backprop_test_factory(mygrad_func=broadcast_to, true_func=np.broadcast_to,
                       num_arrays=1, vary_each_element=True,
                       index_to_arr_shapes={0: hnp.array_shapes(min_dims=0)},
                       kwargs=dict(shape=lambda arr: broadcastable_shape(arr.shape, min_dim=arr.ndim)),
                       assumptions=_is_broadcastable)
def test_broadcast_to_bkwd():
    pass
