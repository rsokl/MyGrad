"""
Utilities for implementing a 2D convolution as a matrix multiplication.

These utilities are adapted from Stanford's CS231n course:
    http://cs231n.github.io/convolutional-networks/
"""
import numpy as np


def get_im2col_indices(w_shape, out_shape, stride):
    """ Returns the advanced indices needed to mutate an array of shape (N, C, H, W) into
        strides of size C*Hf*Hw (N is the number of data samples in the batch). This permits
        a strided 2D convolution to be performed via matrix multiplication.

        Parameters
        ----------
        w_shape : Tuple[int, int, int, int]
            The shape of the filter: (F, C, Hf, Wf). F is the number of filters,
            and C is the channel depth, which must match the channel depth of
            the data.

        out_shape : Tuple[int, int, int, int]
            The shape of the convolution's output: (N, F, Ho, Wo). N is the
            number of pieces of data in the training batch

        stride : Tuple[int, int]
            The step-size with which the filter is placed along the H and W axes
            during the convolution. The tuple indicates (stride-H, stride-W).

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The advanced indices for axes 1, 2, and 3, respectively, of the (N, C, H, W)-shaped
            batch of data. """
    _, c, hf, wf = w_shape
    n, _, ho, wo = out_shape

    i0 = np.repeat(np.arange(hf), wf)
    i0 = np.tile(i0, c)
    i1 = stride[0] * np.repeat(np.arange(ho), wo)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j0 = np.tile(np.arange(wf), hf * c)
    j1 = stride[1] * np.tile(np.arange(wo), ho)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c), hf * wf).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col(x, w_shape, out_shape, padding, stride):
    """ 'Stretch' a (N, C, H, W)-shaped batch of N, CxHxW pieces of data into
        a 2D array so that strided convolutions can be performed over the
        'H' and 'W' dimensions, using filters of channel-depth C, via matrix
        multiplication.

        Parameters
        ----------
        x : numpy.ndarray
            Data batch, shape (N, C, H, W)

        w_shape : Tuple[int, int, int, int]
            The shape of the filter: (F, C, Hf, Wf). F is the number of filters,
            and C is the channel depth, which must match the channel depth of
            the data.

        out_shape : Tuple[int, int, int, int]
            (N, F, H', W')

        padding : Tuple[int, int]
            The number of zeros to be padded to either end of the H-dimension
            and the W-dimension, respectively, of `x`.

        stride : Tuple[int, int]
            The step-size with which the filter is placed along the H and W axes
            during the convolution. The tuple indicates (stride-H, stride-W).

        Returns
        -------
        col_shaped_data : numpy.ndarray
            An array of shape (Hf * Wf * C, N'). Where Hf * Wf * C is the total size
            of a single filter, and N' is the length of the data batch,
            which is strided and repeated such that matrix multiplication with a column
            of filters will replicate a strided convolution.
        """

    _, _, hf, wf = w_shape
    # symmetric 0-padding for H, W dimensions
    axis_pad = tuple((i, i) for i in (0, 0, padding[0], padding[1]))
    x_padded = np.pad(x, axis_pad, mode='constant') if sum(padding) else x

    k, i, j = get_im2col_indices(w_shape, out_shape, stride)
    cols = x_padded[:, k, i, j]

    return cols.transpose(1, 2, 0).reshape(hf * wf * x.shape[1], -1)


def col2im(cols, x_shape, w_shape, out_shape, padding, stride):
    """ The inverse operation of im2col. Convert the result of a strided and tiled 2D convolution,
        produced via matrix multiplication, into the output expected from the standard convolution.

        Parameters
        ----------
        cols : numpy.ndarray
            An array of shape (C * Hf * Wf , N * H' * W'). Hf * Wf * C is the total size
            of a single filter, and N * H' * W' is the total size of the output of a single
            strided convolution of a filter with the entire batch of data.

        x_shape : Tuple[int, int, int, int]
            (N, C, H, W)

        w_shape : Tuple[int, int, int, int]
            (F, C, H, WF)

        out_shape : Tuple[int, int, int, int]
            (N, F, H', W')

        padding : Tuple[int, int]
            The number of zeros to be padded to either end of the H-dimension
            and the W-dimension, respectively, of `x`.

        stride : Tuple[int, int]
            The step-size with which the filter is placed along the H and W axes
            during the convolution. The tuple indicates (stride-H, stride-W).

        Returns
        -------
        col_shaped_data : numpy.ndarray
            Data batch, shape (N, C, H, W)"""
    n, c, h, w = x_shape
    _, _, hf, wf = w_shape
    ph, pw = padding
    x_padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=cols.dtype)
    k, i, j = get_im2col_indices(w_shape, out_shape, stride)
    cols_reshaped = cols.reshape(c * hf * wf, -1, n)  # (C * Hf * Wf, H' * W', N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)  # (N, C * Hf * Wf, H' * W')

    # Accumulate regions of the tiled/strided convolutions into the appropriate
    # sections of x_padded
    # k, i, j all broadcast into (C * Hf * Wf, H' * W'), thus
    # x_padded[:, k, i, j].shape is (N, C * Hf * Wf, H' * W')

    # x_padded is thus tiled/strided by np.add.at to be commensurate with the
    # cols_reshapes and accumulated; this does not mutate its ultimate shape.

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)  # (N, C, H , W)

    return x_padded if not sum(padding) else x_padded[:, :, ph:-ph, pw:-pw]
