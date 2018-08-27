""" Test conv fwd-prop and back-prop for ND convs"""

from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory
from ..utils.numerical_gradient import numerical_gradient_full

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given, assume, settings

from pytest import raises

import numpy as np
from mygrad.nnet.layers import conv_nd
from mygrad import Tensor

from numpy.testing import assert_allclose


def get_outshape(x_shape, w_shape, stride, dilation):
    ''' Compute the shape of the output tensor given an input shape, convolutional
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
    '''
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
    ''' Convolve `input_image` with `conv_filter` at a stride of `stride`.

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
    '''
    conv_shape = conv_filter.shape[1:]
    in_shape = input_image.shape[1:]
    if dilation is None:
        dilation = (1,) * len(stride)
    out_shape = tuple(get_outshape(in_shape, conv_shape, stride, dilation))
    out = np.empty(out_shape, np.float32)
    for ind in np.ndindex(out_shape):
        slices = (slice(None),) + tuple(
            slice(i * s, i * s + w * d, d) for i, w, s, d in zip(ind, conv_shape, stride, dilation))
        out[ind] = np.sum(conv_filter * input_image[slices])
    return out


def conv_bank(input_images, conv_filters, stride, dilation=None):
    ''' Convolve a bank of filters over a stack of images.

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
    '''
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

## Defining Tests


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

    out = np.array([[[-5., -11.]]])
    assert isinstance(o, Tensor)
    assert o.constant is True
    assert o.scalar_only is False
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

    out = np.array([[[[-44.,  -64.],
                      [-84., -104.]]]])
    assert isinstance(o, Tensor)
    assert o.constant is True
    assert o.scalar_only is False
    assert_allclose(actual=o.data, desired=out, err_msg="2d trivial test failed")


def test_bad_conv_shapes():
    x = np.zeros((1, 2, 2, 2))
    w = np.zeros((1, 3, 2, 2))
    with raises(AssertionError):
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


@settings(deadline=2000)
@given(data=st.data(),
       x=hnp.arrays(dtype=float, shape=hnp.array_shapes(max_dims=6, min_dims=3, max_side=15),
                    elements=st.floats(-10, 10)),
       num_filters=st.integers(1, 3))
def test_conv_ND_fwd(data, x, num_filters):
    """ Test convs 1D-4D with various strides and dilations."""
    x = x[0:min(x.shape[0], 3), 0:min(x.shape[1], 3)]
    win_dim = x.ndim - 2
    win_shape = data.draw(st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])), label="win_shape")
    kernels = data.draw(hnp.arrays(dtype=float, shape=(num_filters, x.shape[1], *win_shape),
                                   elements=st.floats(-10, 10)),
                        label="kernels")

    stride = data.draw(st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])), label="stride")

    max_dilation = np.array(x.shape[-win_dim:]) // win_shape
    dilation = data.draw(st.tuples(*(st.integers(1, s) for s in max_dilation)), label="dilation")
    conf = dict(stride=stride, dilation=dilation)

    # skip invalid data/kernel/stride/dilation combinations
    assume(get_outshape(x.shape[2:], kernels.shape[2:], stride, dilation) is not None)
    numpy_conv = conv_bank(x, kernels, **conf)
    mygrad_conv = conv_nd(x, kernels, **conf).data
    assert_allclose(actual=mygrad_conv, desired=numpy_conv, atol=1e-6, rtol=1e-6)


@fwdprop_test_factory(mygrad_func=conv_nd, true_func=conv_bank, num_arrays=2,
                      index_to_arr_shapes={0: (4, 5, 7), 1: (2, 5, 3)},
                      kwargs=dict(stride=(1,), dilation=(1,)))
def test_conv_1d_fwd():
    """ (N=4, C=5, W=7) x (F=2, C=5, Wf=3); stride=1, dilation=1

    Also tests meta properties of conv function - appropriate return type,
    behavior with `constant` arg, etc."""
    pass


def _conv_nd(x, w, stride, dilation=1):
    """ use mygrad-conv_nd forward pass for numerical derivative

        Returns
        -------
        numpy.ndarray"""
    return conv_nd(x, w, stride=stride, dilation=dilation, constant=True).data


settings(deadline=2000)
@given(data=st.data(),
       x=hnp.arrays(dtype=float, shape=hnp.array_shapes(max_dims=5, min_dims=3, max_side=6),
                    elements=st.floats(-10, 10)),
       num_filters=st.integers(1, 3))
def test_conv_ND_bkwd(data, x, num_filters):
    """ Test conv-backprop 1D-3D with various strides and dilations."""
    x = x[0:min(x.shape[0], 1), 0:min(x.shape[1], 1)]
    win_dim = x.ndim - 2
    win_shape = data.draw(st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])), label="win_shape")
    kernels = data.draw(hnp.arrays(dtype=float, shape=(num_filters, x.shape[1], *win_shape),
                                   elements=st.floats(-10, 10)),
                        label="kernels")

    stride = data.draw(st.tuples(*(st.integers(1, s) for s in x.shape[-win_dim:])), label="stride")

    max_dilation = np.array(x.shape[-win_dim:]) // win_shape
    dilation = data.draw(st.tuples(*(st.integers(1, s) for s in max_dilation)), label="dilation")
    conf = dict(stride=stride, dilation=dilation)

    # skip invalid data/kernel/stride/dilation combinations
    assume(get_outshape(x.shape[2:], kernels.shape[2:], stride, dilation) is not None)

    x = Tensor(x)
    kernels = Tensor(kernels)

    out = conv_nd(x, kernels, **conf)
    grad = data.draw(hnp.arrays(shape=out.shape,
                                dtype=float,
                                elements=st.floats(-10, 10),
                                unique=True),
                     label="grad")

    out.backward(grad)
    grads_numerical = numerical_gradient_full(_conv_nd, *(i.data for i in (x, kernels)),
                                              back_grad=grad, kwargs=conf,
                                              as_decimal=False)

    for n, (arr, d_num) in enumerate(zip((x, kernels), grads_numerical)):
        assert_allclose(arr.grad, d_num, atol=1e-4, rtol=1e-4,
                        err_msg="arr-{}: numerical derivative and mygrad derivative do not match".format(n))


@backprop_test_factory(mygrad_func=conv_nd,
                       true_func=_conv_nd,
                       num_arrays=2,
                       index_to_arr_shapes={0: (2, 1, 7), 1: (2, 1, 3)},
                       kwargs={"stride": (1,)},
                       vary_each_element=True, as_decimal=False, atol=1e-4, rtol=1e-4)
def test_conv_1d_bkwd():
    """ (N=2, C=1, W=7) x (F=2, C=1, Wf=3); stride=1, dilation=1

    Also tests meta properties of conv-backprop - appropriate return type,
    behavior with `constant` arg, good behavior of null_gradients, etc."""
    pass

