from mygrad.tensor_base import Tensor
from mygrad.nnet.layers import conv2d, conv_nd
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import fftconvolve, convolve
from itertools import product

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from hypothesis import given

from pytest import raises


def nd_convolve(dat, conv_kernel, stride, outshape=None):
    if np.max(dat.shape) >= 500:
        conv = fftconvolve
    else:
        conv = convolve

    if type(stride) is dict:
        stride_gen = (stride.get(key, 1) for key in range(dat.ndim))
    elif hasattr(stride, '__iter__'):
        stride_gen = stride
    else:
        stride_gen = (stride for i in range(dat.ndim))
    stride = np.fromiter(stride_gen, dtype=int)

    if outshape is None:
        outshape = get_outshape(dat.shape, conv_kernel.shape, stride)

    full_conv = conv(dat, conv_kernel, mode='valid')

    if np.all(stride == 1):
        return full_conv

    # all index positions to down-sample the convolution, given stride > 1
    all_pos = list(zip(*product(*(stride[n]*np.arange(i) for n, i in enumerate(outshape)))))
    out = np.zeros(outshape, dtype=dat.dtype)
    out.flat = full_conv[all_pos]
    return out


def get_outshape(dat_shape, kernel_shape, stride):
    dat_shape = np.array(dat_shape)
    kernel_shape = np.array(kernel_shape)

    if hasattr(stride, '__iter__'):
        stride = np.fromiter(stride, dtype=float)
        assert len(stride) == len(dat_shape), 'The stride iterable must provide a stride value for each dat axis.'
    else:
        stride = float(stride)
    assert len(dat_shape) == len(kernel_shape), "kernel and data must have same number of dimensions"

    outshape = (dat_shape-kernel_shape)/stride+1.
    for num in outshape:
        assert num.is_integer(), num
    outshape = np.rint(outshape).astype(int)

    return outshape


def padder(dat, pad, skip_axes=(0,)):
    assert pad >= 0 and type(pad) == int
    if pad == 0:
        return dat

    if type(skip_axes) == int:
        skip_axes = [skip_axes]
    assert hasattr(skip_axes, '__iter__')
    padding = [(pad, pad) for i in range(dat.ndim)]

    for ax in skip_axes:
        padding[ax] = (0, 0)

    return np.pad(dat, padding, mode='constant').astype(dat.dtype)


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    """

    pad_x = padder(x, conv_param['pad'], skip_axes=[0, 1])
    conv_out_shape = get_outshape(pad_x[0].shape, w[0].shape, conv_param['stride'])
    out = np.zeros((x.shape[0], w.shape[0], conv_out_shape[-2], conv_out_shape[-1]))

    for nk, kernel in enumerate(w):
        # note: we are actually computing a correlation, not a convolution
        conv_kernel = flip_ndarray(kernel)
        for nd, dat in enumerate(pad_x):
            out[nd, nk, :, :] = nd_convolve(dat, conv_kernel, conv_param['stride'], conv_out_shape)
        out[:, nk:nk+1, :, :] += b[nk]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    """

    x, w, b, conv_param = cache

    dx = np.zeros_like(x, dtype=x.dtype)
    dw = np.zeros_like(w, dtype=w.dtype)
    db = np.sum(dout, axis=(0, 2, 3))

    pad = conv_param['pad']
    stride = conv_param['stride']

    npad = np.array([0]+[pad for i in range(x[0].ndim-1)])
    outshape = (np.array(x[0].shape)-np.array(w[0].shape)+2.*npad)/float(stride)+1.
    outshape = np.round(outshape).astype(int)

    # all positions to place the kernel
    all_pos = list(product(*[stride*np.arange(i) for i in outshape]))
    all_slices = [tuple(slice(start, start+w[0].shape[i]) for i,start in enumerate(j)) for j in all_pos]

    if pad:
        pad_ax = [(0, 0)] + [(pad, pad) for i in range(x[0].ndim-1)]

    for nk, kernel in enumerate(w):  # iterate over all kernels
        dx_kernel = np.zeros(x.shape, dtype=x.dtype)
        dkernel = np.zeros_like(kernel, dtype=kernel.dtype)
        for nd, dat in enumerate(x):  # iterate over each piece of data to be convolved
            if pad:
                dat = np.pad(dat, pad_ax, mode='constant').astype(dat.dtype)

            dy = dout[nd, nk][np.newaxis, :, :]
            ddat = np.zeros((x[0].shape[0], x[0].shape[1]+2*pad, x[0].shape[2]+2*pad), dtype=x[0].dtype)

            for i, slices in enumerate(all_slices):
                loc = np.unravel_index(i, outshape)
                dy_val = dy[loc]
                ddat[slices] += dy_val*kernel
                dkernel += dy_val*dat[slices]

            if pad:
                ddat = ddat[:, pad:-pad, pad:-pad]

            dx_kernel[nd] = ddat[:]
        dw[nk:nk+1] = dkernel
        dx += dx_kernel

    return dx, dw, db


def flip_ndarray(x):
    loc = tuple(slice(None, None, -1) for i in range(x.ndim))
    return x[loc]


def test_conv2d_fwd():
    for mem_constr in [True, False]:

        # trivial by-hand test
        # x:
        # [ 1,  2,  3,  4],
        # [ 5,  6,  7,  8],
        # [ 9, 10, 11, 12]]

        # k:
        # [-1, -2],
        # [-3, -4]

        # stride = [1, 2]
        # pad = [0, 0]
        x = np.arange(1, 13).reshape(1, 1, 3, 4)
        k = -1 * np.arange(1, 5).reshape(1, 1, 2, 2)

        o = conv2d(Tensor(x), k, [1, 2], 0)

        out = np.array([[[[-44.,  -64.],
                          [-84., -104.]]]])
        assert isinstance(o, Tensor) and not o.constant and o.scalar_only and np.all(o.data == out)


def test_convnd_fwd():

    # trivial by-hand test
    # x:
    # [ 1,  2,  3,  4],
    # [ 5,  6,  7,  8],
    # [ 9, 10, 11, 12]]

    # k:
    # [-1, -2],
    # [-3, -4]

    # stride = [1, 2]
    # pad = [0, 0]
    x = np.arange(1, 13).reshape(1, 1, 3, 4)
    k = -1 * np.arange(1, 5).reshape(1, 1, 2, 2)

    o = conv_nd(Tensor(x), k, [1, 2], 0)

    out = np.array([[[[-44.,  -64.],
                      [-84., -104.]]]])
    assert isinstance(o, Tensor) and not o.constant and o.scalar_only and np.all(o.data == out)


@given(st.data())
def test_conv2d(data):

    f = data.draw(st.sampled_from([1, 2, 3]))
    c = data.draw(st.sampled_from([1, 2]))

    # w, pad, stride
    ws, pad, stride = data.draw(st.sampled_from([(1, 0, 4), (1, 0, 1), (3, 1, 2), (5, 0, 1)]))

    dat = data.draw(hnp.arrays(shape=(2, c, 5, 5),
                               dtype=float,
                               elements=st.floats(1, 100)))

    w_dat = data.draw(hnp.arrays(shape=(f, c, ws, ws),
                                 dtype=float,
                                 elements=st.floats(1, 100)))

    x = Tensor(dat)
    w = Tensor(w_dat)
    f = conv2d(x, w, stride, pad)

    b = np.zeros((w.shape[0],))
    out, _ = conv_forward_naive(dat, w_dat, b, {'stride': stride, 'pad': pad})

    assert_allclose(f.data, out)

    dout = data.draw(hnp.arrays(shape=f.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    f.backward(dout)
    dx, dw, db = conv_backward_naive(dout, _)
    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
    assert_allclose(w.grad, dw, atol=1e-5, rtol=1e-5)


@given(st.data())
def test_convnd(data):

    # test 2D
    num_conv = 2
    f = data.draw(st.sampled_from([1, 2, 3]))
    c = data.draw(st.sampled_from([1, 2]))

    # w, pad, stride
    ws, pad, stride = data.draw(st.sampled_from([(1, 0, 4), (1, 0, 1), (3, 1, 2), (5, 0, 1)]))

    dat = data.draw(hnp.arrays(shape=(2, c) + (5,)*num_conv,
                               dtype=float,
                               elements=st.floats(1, 100)))

    w_dat = data.draw(hnp.arrays(shape=(f, c) + (ws,)*num_conv,
                                 dtype=float,
                                 elements=st.floats(1, 100)))

    x = Tensor(dat)
    w = Tensor(w_dat)
    f = conv_nd(x, w, stride, pad)

    b = np.zeros((w.shape[0],))
    out, _ = conv_forward_naive(dat, w_dat, b, {'stride': stride, 'pad': pad})

    assert_allclose(f.data, out)

    dout = data.draw(hnp.arrays(shape=f.shape,
                                dtype=float,
                                elements=st.floats(-100, 100)))

    f.backward(dout)
    dx, dw, db = conv_backward_naive(dout, _)
    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
    assert_allclose(w.grad, dw, atol=1e-5, rtol=1e-5)


def test_bad_conv_shapes():
    x = np.zeros((1, 2, 2, 2))
    w = np.zeros((1, 3, 2, 2))
    with raises(AssertionError):
        conv2d(x, w, 1, 0)  # mismatched channels

    w = np.zeros((1, 2, 3, 2))
    with raises(ValueError):
        conv2d(x, w, 1, 0)  # large filter

    w = np.zeros((1, 2, 2, 2))
    with raises(AssertionError):
        conv2d(x, w, 0, 0)  # bad stride

    with raises(AssertionError):
        conv2d(x, w, [1, 2, 3], 0)  # bad stride

    with raises(AssertionError):
        conv2d(x, w, 1, -1)  # bad pad

    with raises(AssertionError):
        conv2d(x, w, 1, [1, 2, 3])  # bad pad

    with raises(ValueError):
        conv2d(x, w, 3, 1)  # shape mismatch
