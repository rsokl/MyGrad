""" Test conv fwd-prop and back-prop for ND convs"""

from typing import Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from numpy.testing import assert_allclose
from pytest import raises

import mygrad as mg
from mygrad import Tensor
from mygrad.nnet.layers import conv_nd

from ...utils.numerical_gradient import numerical_gradient_full
from ...wrappers.uber import backprop_test_factory, fwdprop_test_factory


@pytest.mark.parametrize(
    "shapes",
    [  # x has too few dims
        st.tuples(
            hnp.array_shapes(min_dims=0, max_dims=2), hnp.array_shapes(min_dims=3)
        ),
        # x.ndim != k.ndim
        hnp.array_shapes(min_dims=3).flatmap(
            lambda x: st.tuples(
                st.just(x),
                hnp.array_shapes(min_dims=2).filter(lambda s: len(s) != len(x)),
            )
        ),
        # channel sizes don't match
        hnp.array_shapes(min_dims=3).flatmap(
            lambda x: st.tuples(
                st.just(x),
                hnp.array_shapes(min_dims=len(x), max_dims=len(x)).filter(
                    lambda s: s[1] != x[1]
                ),
            )
        ),
    ],
)
@given(data=st.data())
def test_input_validation(
    shapes: st.SearchStrategy[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    data: st.DataObject,
):
    x_shape, k_shape = data.draw(shapes, label="x_shape, k_shape")
    x = mg.zeros(x_shape, dtype="float")
    k = mg.zeros(k_shape, dtype="float")

    with raises(ValueError):
        conv_nd(x, k, stride=1)


def get_outshape(x_shape, w_shape, stride, dilation):
    """Compute the shape of the output tensor given an input shape, convolutional
    filter shape, and stride.

    Parameters
    ----------
    x_shape : Tuple[int, ...]
        The shape of the input tensor.

    w_shape : Tuple[int, ...]
        The shape of the convolutional filter.

    stride : Tuple[int, ...]
        The stride at which to apply the convolutional filter to the input.

    dilation : Tuple[int, ...]
        The dilation used to form each window over the data.

    Returns
    -------
    numpy.ndarray[int], shape=(num_conv,)
        The shape of the output tensor resulting from convolving a tensor of shape `x_shape`
        with a tensor of shape `w_shape`.

        Returns `None` if an invalid combination of shapes are provided.
    """
    x_shape = np.array(x_shape)
    w_shape = np.array(w_shape)
    stride = np.array(stride)
    dilation = np.array(dilation)
    out_shape = (x_shape - ((w_shape - 1) * dilation + 1)) / stride + 1

    if not all(i.is_integer() and i > 0 for i in out_shape):
        msg = "Stride and kernel dimensions are incompatible: \n"
        msg += "Input dimensions: {}\n".format(tuple(x_shape))
        msg += "Stride dimensions: {}\n".format(tuple(stride))
        msg += "Kernel dimensions: {}\n".format(tuple(w_shape))
        msg += "Dilation dimensions: {}\n".format(tuple(dilation))
        return None
    return out_shape.astype(np.int32)


def convolve_numpy(input_image, conv_filter, stride, dilation=None):
    """Convolve `input_image` with `conv_filter` at a stride of `stride`.

    Parameters
    ----------
    input_image : numpy.ndarray, shape=(C, H, ...)
        The input over which to perform convolution.

    conv_filter : numpy.ndarray, shape=(C, Hf, ...)
        The convolutional filter to slide across the image.

    stride : Sequence[int]
        The stride at which to apply `conv_filter` across `input_image`.

    Returns
    -------
    numpy.ndarray, shape=(H', ...)
        The result of convolving `input_image` with `conv_filter` at a stride of `stride`,
        where (H', W') is the result of `get_outshape`.
    """
    conv_shape = conv_filter.shape[1:]
    in_shape = input_image.shape[1:]
    if dilation is None:
        dilation = (1,) * len(stride)
    out_shape = tuple(get_outshape(in_shape, conv_shape, stride, dilation))
    out = np.empty(out_shape, np.float32)
    for ind in np.ndindex(out_shape):
        slices = (slice(None),) + tuple(
            slice(i * s, i * s + w * d, d)
            for i, w, s, d in zip(ind, conv_shape, stride, dilation)
        )
        out[ind] = np.sum(conv_filter * input_image[slices])
    return out


def conv_bank(input_images, conv_filters, stride, dilation=None, padding=tuple()):
    """Convolve a bank of filters over a stack of images.

    Parameters
    ----------
    input_images : numpy.ndarray, shape=(N, C, H, ...)
        The images over which to convolve our filters.

    conv_filters : numpy.ndarray, shape=(K, C, Hf, ...)
        The convolutional filters to apply to the images.

    stride : Sequence[int]
        The stride at which to apply each filter to the images.

    dilation : Sequence[int]

    Returns
    -------
    numpy.ndarray, shape=(N, K, H', ...)
        The result of convolving `input_image` with `conv_filter` at a stride of `stride`,
        where (H', ...) is the result of `get_outshape`.
    """

    if isinstance(padding, int):
        padding = (padding,) * (input_images.ndim - 2)
    if sum(padding):
        # symmetric 0-padding for X0, X1, ... dimensions
        axis_pad = tuple((i, i) for i in (0, 0, *padding))
        input_images = np.pad(input_images, axis_pad, mode="constant")

    img_shape = input_images.shape[2:]
    conv_shape = conv_filters.shape[2:]

    if dilation is None:
        dilation = (1,) * len(stride)
    out_shape = get_outshape(img_shape, conv_shape, stride, dilation)

    out = np.empty((len(input_images), len(conv_filters), *out_shape))
    for i, image in enumerate(input_images):
        for j, conv in enumerate(conv_filters):
            out[i, j] = convolve_numpy(image, conv, stride, dilation)
    return out


def test_convnd_fwd_trivial():
    # trivial by-hand test: 1-dimensional conv
    # x:
    # [ 1,  2,  3,  4]

    # k:
    # [-1, -2],

    # stride = (2,)
    x = Tensor(np.arange(1, 5).reshape(1, 1, 4).astype(float))
    k = Tensor(-1 * np.arange(1, 3).reshape(1, 1, 2).astype(float))

    o = conv_nd(x, k, stride=(2,), constant=True)

    out = np.array([[[-5.0, -11.0]]])
    assert isinstance(o, Tensor)
    assert o.constant is True
    assert_allclose(actual=o.data, desired=out, err_msg="1d trivial test failed")

    # trivial by-hand test: 2-dimensional conv
    # x:
    # [ 1,  2,  3,  4],
    # [ 5,  6,  7,  8],
    # [ 9, 10, 11, 12]]

    # k:
    # [-1, -2],
    # [-3, -4]

    # stride = (1, 2)
    x = Tensor(np.arange(1, 13).reshape(1, 1, 3, 4).astype(float))
    k = Tensor(-1 * np.arange(1, 5).reshape(1, 1, 2, 2).astype(float))

    o = conv_nd(Tensor(x), k, stride=(1, 2), constant=True)

    out = np.array([[[[-44.0, -64.0], [-84.0, -104.0]]]])
    assert isinstance(o, Tensor)
    assert o.constant is True
    assert_allclose(actual=o.data, desired=out, err_msg="2d trivial test failed")


def test_bad_conv_shapes():
    x = np.zeros((1, 2, 2, 2))
    w = np.zeros((1, 3, 2, 2))
    with raises(ValueError):
        conv_nd(x, w, stride=1, padding=0)  # mismatched channels

    w = np.zeros((1, 2, 3, 2))
    with raises(ValueError):
        conv_nd(x, w, stride=1, padding=0)  # large filter

    w = np.zeros((1, 2, 2, 2))
    with raises(AssertionError):
        conv_nd(x, w, stride=0, padding=0)  # bad stride

    with raises(AssertionError):
        conv_nd(x, w, stride=[1, 2, 3])  # bad stride

    with raises(AssertionError):
        conv_nd(x, w, stride=1, padding=-1)  # bad pad

    with raises(AssertionError):
        conv_nd(x, w, stride=1, padding=[1, 2, 3])  # bad pad

    with raises(ValueError):
        conv_nd(x, w, stride=3, padding=1)  # shape mismatch


@settings(deadline=None)
@given(ndim=st.integers(1, 4), data=st.data())
def test_padding(ndim: int, data: st.DataObject):
    """Ensure that convolving a padding-only image with a commensurate kernel yields the single entry: 0"""
    padding = data.draw(
        st.integers(1, 3) | st.tuples(*[st.integers(1, 3)] * ndim), label="padding"
    )
    x = Tensor(
        data.draw(
            hnp.arrays(shape=(1, 1) + (0,) * ndim, dtype=float, elements=st.floats()),
            label="x",
        )
    )
    pad_tuple = padding if isinstance(padding, tuple) else (padding,) * ndim
    kernel = data.draw(
        hnp.arrays(
            shape=(1, 1) + tuple(2 * p for p in pad_tuple),
            dtype=float,
            elements=st.floats(allow_nan=False, allow_infinity=False),
        )
    )
    out = conv_nd(x, kernel, padding=padding, stride=1)
    assert out.shape == (1,) * x.ndim
    assert out.item() == 0.0

    out.sum().backward()
    assert x.grad.shape == x.shape


@fwdprop_test_factory(
    mygrad_func=conv_nd,
    true_func=conv_bank,
    num_arrays=2,
    index_to_arr_shapes={0: (4, 5, 7), 1: (2, 5, 3)},
    kwargs=dict(stride=(1,), dilation=(1,)),
    index_to_bnds={0: (-10, 10), 1: (-10, 10)},
)
def test_conv_1d_fwd():
    """(N=4, C=5, W=7) x (F=2, C=5, Wf=3); stride=1, dilation=1

    Also tests meta properties of conv function - appropriate return type,
    behavior with `constant` arg, etc."""


@mg.no_autodiff
def _conv_nd(x, w, stride, dilation=1, padding=0) -> np.ndarray:
    """use mygrad-conv_nd forward pass for numerical derivative

    Returns
    -------
    numpy.ndarray"""
    return mg.asarray(
        conv_nd(x, w, stride=stride, dilation=dilation, padding=padding, constant=True)
    )


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=conv_nd,
    true_func=_conv_nd,
    num_arrays=2,
    index_to_arr_shapes={0: (2, 1, 7), 1: (2, 1, 3)},
    kwargs={"stride": (1,)},
    index_to_bnds={0: (-10, 10), 1: (-10, 10)},
    vary_each_element=True,
)
def test_conv_1d_bkwd():
    """(N=2, C=1, W=7) x (F=2, C=1, Wf=3); stride=1, dilation=1

    Also tests meta properties of conv-backprop - appropriate return type,
    behavior with `constant` arg, good behavior of null_gradients, etc."""


@settings(deadline=None, suppress_health_check=(HealthCheck.filter_too_much,))
@given(
    data=st.data(),
    shape=hnp.array_shapes(min_dims=1, max_dims=3, max_side=10),
    num_filters=st.integers(1, 3),
    num_batch=st.integers(1, 3),
    num_channel=st.integers(1, 3),
)
def test_conv_ND_fwd(data, shape, num_filters, num_batch, num_channel):
    img_shape = (num_batch, num_channel) + shape

    padding = data.draw(
        st.integers(0, 2) | st.tuples(*[st.integers(0, 2)] * len(shape)),
        label="padding",
    )

    if isinstance(padding, tuple):
        shape = tuple(s + 2 * p for s, p in zip(shape, padding))
    else:
        shape = tuple(s + 2 * padding for s in shape)

    win_dim = len(shape)
    shape = (num_batch, num_channel) + shape
    win_shape = data.draw(
        st.tuples(*(st.integers(1, s) for s in shape[-win_dim:])), label="win_shape"
    )
    kernel_shape = (num_filters, shape[1], *win_shape)
    stride = data.draw(
        st.tuples(*(st.integers(1, s) for s in shape[-win_dim:])), label="stride"
    )
    max_dilation = np.array(shape[-win_dim:]) // win_shape
    dilation = data.draw(
        st.tuples(*(st.integers(1, s) for s in max_dilation)), label="dilation"
    )
    conf = dict(stride=stride, dilation=dilation, padding=padding)

    # skip invalid data/kernel/stride/dilation combinations
    assume(get_outshape(shape[2:], kernel_shape[2:], stride, dilation) is not None)

    kernels = data.draw(
        hnp.arrays(dtype=float, shape=kernel_shape, elements=st.floats(-10, 10)),
        label="kernels",
    )
    x = data.draw(
        hnp.arrays(dtype=float, shape=img_shape, elements=st.floats(-10, 10)), label="x"
    )

    mygrad_conv = conv_nd(x, kernels, **conf).data
    numpy_conv = conv_bank(x, kernels, **conf)
    assert_allclose(actual=mygrad_conv, desired=numpy_conv, atol=1e-6, rtol=1e-6)


@settings(deadline=None, suppress_health_check=(HealthCheck.filter_too_much,))
@given(
    data=st.data(),
    shape=hnp.array_shapes(min_dims=1, max_dims=3, max_side=6),
    num_filters=st.integers(1, 3),
    num_batch=st.integers(1, 3),
    num_channel=st.integers(1, 3),
)
def test_conv_ND_bkwd(data, shape, num_filters, num_batch, num_channel):
    """Test conv-backprop 1D-3D with various strides and dilations."""
    img_shape = (num_batch, num_channel) + shape

    padding = data.draw(
        st.integers(0, 2) | st.tuples(*[st.integers(0, 2)] * len(shape)),
        label="padding",
    )

    if isinstance(padding, tuple):
        shape = tuple(s + 2 * p for s, p in zip(shape, padding))
    else:
        shape = tuple(s + 2 * padding for s in shape)

    win_dim = len(shape)
    shape = (num_batch, num_channel) + shape
    win_shape = data.draw(
        st.tuples(*(st.integers(1, s) for s in shape[-win_dim:])), label="win_shape"
    )
    kernel_shape = (num_filters, shape[1], *win_shape)

    stride = data.draw(
        st.tuples(*(st.integers(1, s) for s in shape[-win_dim:])), label="stride"
    )

    max_dilation = np.array(shape[-win_dim:]) // win_shape
    dilation = data.draw(
        st.tuples(*(st.integers(1, s) for s in max_dilation)), label="dilation"
    )
    conf = dict(stride=stride, dilation=dilation, padding=padding)

    # skip invalid data/kernel/stride/dilation combinations
    assume(get_outshape(shape[2:], kernel_shape[2:], stride, dilation) is not None)

    kernels = data.draw(
        hnp.arrays(dtype=float, shape=kernel_shape, elements=st.floats(-10, 10)),
        label="kernels",
    )
    x = data.draw(
        hnp.arrays(dtype=float, shape=img_shape, elements=st.floats(-10, 10)), label="x"
    )

    x = Tensor(x)
    kernels = Tensor(kernels)

    out = conv_nd(x, kernels, **conf)
    grad = data.draw(
        hnp.arrays(
            shape=out.shape, dtype=float, elements=st.floats(-10, 10), unique=True
        ),
        label="grad",
    )

    out.backward(grad)
    grads_numerical = numerical_gradient_full(
        _conv_nd, *(i.data for i in (x, kernels)), back_grad=grad, kwargs=conf
    )

    for n, (arr, d_num) in enumerate(zip((x, kernels), grads_numerical)):
        assert_allclose(
            arr.grad,
            d_num,
            atol=1e-4,
            rtol=1e-4,
            err_msg="arr-{}: numerical derivative and mygrad derivative do not match".format(
                n
            ),
        )
