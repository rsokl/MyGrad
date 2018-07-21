""" Test conv fwd-prop and back-prop for 1D, 2D, and 3D"""

from ..wrappers.uber import backprop_test_factory

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given

import numpy as np
from mygrad.nnet.layers import conv_nd

from numpy.testing import assert_allclose


def get_outshape(x_shape, w_shape, stride):
    ''' Compute the shape of the output tensor given an input shape, convolutional
    filter shape, and stride.

    Parameters
    ----------
    x_shape : Tuple[int, int]
        The shape of the input tensor.

    w_shape : Tuple[int, int]
        The shape of the convolutional filter.

    stride : Tuple[int, int]
        The stride at which to apply the convolutional filter to the input.

    Returns
    -------
    numpy.ndarray[int], shape=(2,)
        The shape of the output tensor resulting from convolving a tensor of shape `x_shape`
        with a tensor of shape `w_shape`.
    '''
    x_shape = np.array(x_shape)
    w_shape = np.array(w_shape)
    stride = np.array(stride)

    out_shape = (x_shape - w_shape) / stride + 1

    if not all(i.is_integer() and i > 0 for i in out_shape):
        msg = "Stride and kernel dimensions are incompatible: \n"
        msg += "Input dimensions: {}\n".format(tuple(x_shape))
        msg += "Stride dimensions: {}\n".format(tuple(stride))
        msg += "Kernel dimensions: {}\n".format(tuple(w_shape))
        raise ValueError(msg)
    return out_shape.astype(np.int32)


def convolve_numpy(input_image, conv_filter, stride):
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
    out_shape = tuple(get_outshape(in_shape, conv_shape, stride))
    out = np.empty(out_shape, np.float32)
    for ind in np.ndindex(out_shape):
        slices = (slice(None),) + tuple(slice(i * s, i * s + w) for i, w, s in zip(ind, conv_shape, stride))
        out[ind] = np.sum(conv_filter * input_image[slices])
    return out


def conv_bank(input_images, conv_filters, stride):
    ''' Convolve a bank of filters over a stack of images.

    Parameters
    ----------
    input_images : numpy.ndarray, shape=(N, C, H, ...)
        The images over which to convolve our filters.

    conv_filters : numpy.ndarray, shape=(K, C, Hf, ...)
        The convolutional filters to apply to the images.

    stride : Sequence[int]
        The stride at which to apply each filter to the images.

    Returns
    -------
    numpy.ndarray, shape=(N, K, H', ...)
        The result of convolving `input_image` with `conv_filter` at a stride of `stride`,
        where (H', ...) is the result of `get_outshape`.
    '''
    img_shape = input_images.shape[2:]
    conv_shape = conv_filters.shape[2:]
    out_shape = get_outshape(img_shape, conv_shape, stride)

    out = np.empty((len(input_images), len(conv_filters), *out_shape))
    for i, image in enumerate(input_images):
        for j, conv in enumerate(conv_filters):
            out[i, j] = convolve_numpy(image, conv, stride)
    return out


@given(x=hnp.arrays(shape=(4, 5, 7),
                    dtype=float,
                    elements=st.floats(-10, 10)),
       w=hnp.arrays(shape=(2, 5, 3),
                    dtype=float,
                    elements=st.floats(-10, 10)))
def test_conv1_1d_fwd(x, w):
    """ (N=4, C=5, W=7) x (F=2, C=5, Wf=3); stride=1"""
    numpy_conv = conv_bank(x, w, stride=(1,))

    mygrad_conv = conv_nd(x, w, stride=(1,)).data
    assert_allclose(actual=mygrad_conv, desired=numpy_conv)


@given(x=hnp.arrays(shape=(1, 2, 7, 6),
                    dtype=float,
                    elements=st.floats(-10, 10)),
       w=hnp.arrays(shape=(2, 2, 3, 3),
                    dtype=float,
                    elements=st.floats(-10, 10)))
def test_conv1_2d_fwd(x, w):
    """ (N=1, C=2, H=7, W=6) x (F=2, C=2, Hf=3, Wf=3); stride=(2, 1)"""
    stride = (2, 1)
    numpy_conv = conv_bank(x, w, stride=stride)
    mygrad_conv = conv_nd(x, w, stride=stride).data
    assert_allclose(actual=mygrad_conv, desired=numpy_conv)


@given(x=hnp.arrays(shape=(1, 2, 7, 6, 4),
                    dtype=float,
                    elements=st.floats(-10, 10)),
       w=hnp.arrays(shape=(2, 2, 3, 2, 1),
                    dtype=float,
                    elements=st.floats(-10, 10)))
def test_conv1_3d_fwd(x, w):
    """ (N=1, C=2, 7, 6, 4) x (F=2, C=2, 3, 2, 1); stride=(2, 1, 3)"""
    stride = (2, 1, 3)
    numpy_conv = conv_bank(x, w, stride=stride)
    mygrad_conv = conv_nd(x, w, stride=stride).data
    assert_allclose(actual=mygrad_conv, desired=numpy_conv)


def _conv_nd(x, w, stride):
    """ use mygrad-conv_nd forward pass for numerical derivative

        Returns
        -------
        numpy.ndarray"""
    return conv_nd(x, w, stride=stride, constant=True).data


@backprop_test_factory(mygrad_func=conv_nd,
                       true_func=_conv_nd,
                       num_arrays=2,
                       index_to_arr_shapes={0: (1, 2, 7), 1: (1, 2, 3)},
                       kwargs={"stride": (1,)},
                       vary_each_element=True, as_decimal=False, atol=1e-4, rtol=1e-4)
def test_conv_1d_bkwd():
    pass


@backprop_test_factory(mygrad_func=conv_nd,
                       true_func=_conv_nd,
                       num_arrays=2,
                       index_to_arr_shapes={0: (1, 2, 7), 1: (2, 2, 3)},
                       kwargs={"stride": (2,)},
                       vary_each_element=True, as_decimal=False, atol=1e-4, rtol=1e-4)
def test_conv_1d_bkwd2():
    pass


@backprop_test_factory(mygrad_func=conv_nd,
                       true_func=_conv_nd,
                       num_arrays=2,
                       index_to_arr_shapes={0: (1, 1, 7, 6), 1: (2, 1, 3, 3)},
                       kwargs={"stride": (2, 1)},
                       vary_each_element=True, as_decimal=False, atol=1e-4, rtol=1e-4)
def test_conv_2d_bkwd():
    pass


@backprop_test_factory(mygrad_func=conv_nd,
                       true_func=_conv_nd,
                       num_arrays=2,
                       index_to_arr_shapes={0: (1, 1, 2, 2, 4), 1: (1, 1, 2, 2, 2)},
                       kwargs={"stride": (1, 1, 2)},
                       vary_each_element=True, as_decimal=False, atol=1e-3, rtol=1e-3)
def test_conv_3d_bkwd():
    pass