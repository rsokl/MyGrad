
import numpy as np


def sliding_window_view(arr, window_shape, step, dilation=None):
    """ Create a sliding window view over the trailing dimesnions of an array.
        No copy is made.

        The window is applied only to valid regions of `arr`, but is applied geedily.

        See Notes section for details.

        Parameters
        ----------
        arr : numpy.ndarray, shape=(..., [x, (...), z])
            C-ordered array over which sliding view-window is applied along the trailing
            dimensions [x, ..., z], as determined by the length of `window_shape`.

        window_shape : Sequence[int]
            Specifies the shape of the view-window: [Wx, (...), Wz].
            The length of `window_shape` determines the length of [x, (...) , z]

        step : Union[int, Sequence[int]]
            The step sized used along the [x, (...), z] dimensions: [Sx, (...), Sz].
            If a single integer is specified, a uniform step size is used.

        dilation : Optional[Sequence[int]]
            The dilation factor used along the [x, (...), z] directions: [Dx, (...), Dz].
            If no value is specified, a dilation factor of 1 is used along each direction.
            Dilation specifies the step size used when filling the window's elements

        Returns
        -------
        A contiguous view of `arr`, of shape ([X, (...), Z], ..., [Wx, (...), Wz]), where
        [X, ..., Z] is the shape of the grid on which the window was applied. See Notes
        sections for more details.

        Notes
        -----
        Window placement:
            Given a dimension of size x, with a window of size W along this dimension, applied
            with stride S and dilation D, the window will be applied
                                      X = (x - (W - 1) * D + 1) // S + 1
            number of times along that dimension.

        Interpreting output:
            In general, given an array `arr` of shape (..., x, (...), z), and
                `out = sliding_window_view(arr, window_shape=[Wx, (...), Wz], step=[Sx, (...), Sz])`
            the indexing `out` with [ix, (...), iz] produces the following view of x:

                `out[ix, (...), iz] ==
                    x[..., ix*Sx:(ix*Sx + Wx*Dx):Dx, (...), iz*Sz:(iz*Sz + Wz*Dz):Dz]`

            For example, suppose `arr` is an array of shape (10, 12, 6). Specifying sliding
            window of shape (3, 3) with step size (2, 2), dilation (2, 1) will create the view:

                            [[arr[:,  0:6:2, 0:3], arr[:,   0:6:3, 3:6]]
                             [arr[:, 6:12:2, 0:3], arr[:, 6:12:12, 3:6]]]

            producing a view of shape (2, 2, 10, 3, 3) in total.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.arange(36).reshape(6, 6)
        >>> x
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23],
               [24, 25, 26, 27, 28, 29],
               [30, 31, 32, 33, 34, 35]])

        Apply an 3x2 window with step-sizes of (2, 2). This results in
        the window being placed twice along axis-0 and three times along axis-1.
        >>> y = sliding_window_view(x, step=(2, 2), window_shape=(3, 2))
        >>> y.shape
        (2, 3, 3, 2)

        # window applied at (0, 0)
        >>> y[0, 0]
        array([[ 0,  1],
               [ 6,  7],
               [12, 13]])

        # window applied at (2, 0)
        >>> y[1, 0]
        array([[12, 13],
               [18, 19],
               [24, 25]])

        # window applied at (0, 2)
        >>> y[0, 1]
        array([[ 2,  3],
               [ 8,  9],
               [14, 15]])

        >>> i, j = np.random.randint(0, 2, size=2)
        >>> wx, wy = (2, 2)
        >>> sx, sy = (2, 2)
        >>> np.all(y[i, j] == x[..., i*sx:(i*sx + wx), j*sy:(j*sy + wy)])
        True
        """

    from numbers import Integral
    from numpy.lib.stride_tricks import as_strided
    import numpy as np

    step = tuple(int(step) for i in range(len(window_shape))) if isinstance(step, Integral) else tuple(step)
    assert all(isinstance(i, Integral) and i > 0 for i in step), "`step` be a sequence of positive integers"

    window_shape = tuple(window_shape)
    if not all(isinstance(i, Integral) and i > 0 for i in window_shape):
        msg = "`window_shape` be a sequence of positive integers"
        raise AssertionError(msg)

    if len(window_shape) > arr.ndim:
        msg = """ `window_shape` cannot specify more values than `arr.ndim`."""
        raise AssertionError(msg)

    if any(i > j for i, j in zip(window_shape[::-1], arr.shape[::-1])):
        msg = """ The window must fit within the trailing dimensions of `arr`."""
        raise AssertionError(msg)

    if dilation is None:
        dilation = np.ones((len(window_shape),), dtype=int)
    else:
        if isinstance(dilation, Integral):
            dilation = tuple(int(dilation) for i in range(len(window_shape)))
        else:
            np.asarray(dilation)
            assert all(isinstance(i, Integral) for i in dilation)
            if any(w * d > s for w, d, s in zip(window_shape[::-1], dilation[::-1], arr.shape[::-1])):
                msg = """ The dilated window must fit within the trailing dimensions of `arr`."""
                raise AssertionError(msg)

    step = np.array(step)  # (Sx, ..., Sz)
    window_shape = np.array(window_shape)  # (Wx, ..., Wz)
    in_shape = np.array(arr.shape[-len(step):])  # (x, ... , z)
    nbyte = arr.strides[-1]  # size, in bytes, of element in `arr`

    # per-byte strides required to fill a window
    win_stride = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)

    # per-byte strides required to advance the window
    step_stride = tuple(win_stride[-len(step):] * step)

    # update win_stride to accommodate dilation
    win_stride = np.array(win_stride)
    win_stride[-len(step):] *= dilation
    win_stride = tuple(win_stride)

    # tuple of bytes to step to traverse corresponding dimensions of view
    # see: 'internal memory layout of an ndarray'
    stride = tuple(int(nbyte * i) for i in step_stride + win_stride)

    # number of window placements along x-dim: X = (x - (Wx - 1)*Dx + 1) // Sx + 1
    out_shape = tuple((in_shape - ((window_shape - 1) * dilation + 1)) // step + 1)

    # ([X, (...), Z], ..., [Wx, (...), Wz])
    out_shape = out_shape + arr.shape[:-len(step)] + tuple(window_shape)
    out_shape = tuple(int(i) for i in out_shape)

    return as_strided(arr, shape=out_shape, strides=stride, writeable=False)



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
