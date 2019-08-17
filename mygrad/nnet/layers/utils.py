from numbers import Integral

import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window_view(arr, window_shape, step, dilation=None):
    """ Create a sliding window view over the trailing dimensions of an array.
        No copy is made.

        The window is applied only to valid regions of ``arr``, but is applied geedily.

        See Notes section for details.

        Parameters
        ----------
        arr : numpy.ndarray, shape=(..., [x, (...), z])
            C-contiguous array over which sliding view-window is applied along the trailing
            dimensions ``[x, ..., z]``, as determined by the length of ``window_shape``.

            If ``arr`` is not C-contiguous, it will be replaced by ``numpy.ascontiguousarray(arr)``

        window_shape : Sequence[int]
            Specifies the shape of the view-window: ``[Wx, (...), Wz]``.
            The length of `window_shape` determines the length of ``[x, (...) , z]``.

        step : Union[int, Sequence[int]]
            The step sized used along the ``[x, (...), z]`` dimensions: ``[Sx, (...), Sz]``.
            If a single integer is specified, a uniform step size is used.

        dilation : Optional[Union[int, Sequence[int]]]
            The dilation factor used along the ``[x, (...), z]`` directions: ``[Dx, (...), Dz]``.
            If no value is specified, a dilation factor of 1 is used along each direction.
            Dilation specifies the step size used when filling the window's elements

        Returns
        -------
        numpy.ndarray
            A contiguous view of ``arr``, of shape ``([X, (...), Z], ..., [Wx, (...), Wz])``, where
            ``[X, ..., Z]`` is the shape of the grid on which the window was applied. See Notes
            sections for more details.

        Raises
        ------
        ValueError, TypeError
            Invalid step-size, window shape, or dilation

        Notes
        -----
        Window placement:
            Given a dimension of size x, with a window of size W along this dimension, applied
            with stride S and dilation D, the window will be applied::
                                      X = (x - (W - 1) * D + 1) // S + 1
            number of times along that dimension.

        Interpreting output:
            In general, given an array ``arr`` of shape (..., x, (...), z), and::

                out = sliding_window_view(arr, window_shape=[Wx, (...), Wz], step=[Sx, (...), Sz])

            then indexing ``out`` with ``[ix, (...), iz]`` produces the following view of ``x``::

                out[ix, (...), iz] ==
                    x[..., ix*Sx:(ix*Sx + Wx*Dx):Dx, (...), iz*Sz:(iz*Sz + Wz*Dz):Dz]

            For example, suppose ``arr`` is an array of shape-(10, 12, 6). Specifying sliding
            window of shape ``(3, 3)`` with step size ``(2, 2)``, dilation ``(2, 1)`` will create the view::

                            [[arr[:,  0:6:2, 0:3], arr[:,   0:6:3, 3:6]]
                             [arr[:, 6:12:2, 0:3], arr[:, 6:12:12, 3:6]]]

            producing a view of shape ``(2, 2, 10, 3, 3)`` in total.

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

        window applied at (0, 0)

        >>> y[0, 0]
        array([[ 0,  1],
               [ 6,  7],
               [12, 13]])

        window applied at (2, 0)

        >>> y[1, 0]
        array([[12, 13],
               [18, 19],
               [24, 25]])

        window applied at (0, 2)

        >>> y[0, 1]
        array([[ 2,  3],
               [ 8,  9],
               [14, 15]])

        verify that an element in this window-view is correct

        >>> i, j = np.random.randint(0, 2, size=2)
        >>> wx, wy = (2, 2)
        >>> sx, sy = (2, 2)
        >>> np.all(y[i, j] == x[..., i*sx:(i*sx + wx), j*sy:(j*sy + wy)])
        True
        """

    if not hasattr(window_shape, "__iter__"):
        raise TypeError(
            "`window_shape` be a sequence of positive integers, got: {}".format(
                window_shape
            )
        )
    window_shape = tuple(window_shape)
    if not all(isinstance(i, Integral) and i > 0 for i in window_shape):
        raise TypeError(
            "`window_shape` be a sequence of positive integers, "
            "got: {}".format(window_shape)
        )

    if len(window_shape) > arr.ndim:
        raise ValueError(
            "`window_shape` ({}) cannot specify more values than "
            "`arr.ndim` ({}).".format(window_shape, arr.ndim)
        )

    if not isinstance(step, Integral) and not hasattr(step, "__iter__"):
        raise TypeError(
            "`step` be a positive integer or a sequence of positive "
            "integers, got: {}".format(step)
        )

    step = (
        (int(step),) * len(window_shape) if isinstance(step, Integral) else tuple(step)
    )

    if not all(isinstance(i, Integral) and i > 0 for i in step):
        raise ValueError(
            "`step` be a positive integer or a sequence of positive "
            "integers, got: {}".format(step)
        )

    if any(i > j for i, j in zip(window_shape[::-1], arr.shape[::-1])):
        raise ValueError(
            "Each size of the window-shape must fit within the trailing "
            "dimensions of `arr`."
            "{} does not fit in {}".format(
                window_shape, arr.shape[-len(window_shape) :]
            )
        )

    if (
        dilation is not None
        and not isinstance(dilation, Integral)
        and not hasattr(dilation, "__iter__")
    ):
        raise TypeError(
            "`dilation` be None, a positive integer, or a sequence of "
            "positive integers, got: {}".format(dilation)
        )
    if dilation is None:
        dilation = np.ones((len(window_shape),), dtype=int)
    else:
        if isinstance(dilation, Integral):
            dilation = np.full((len(window_shape),), fill_value=dilation, dtype=int)
        else:
            np.asarray(dilation)

        if not all(isinstance(i, Integral) and i > 0 for i in dilation) or len(
            dilation
        ) != len(window_shape):
            raise ValueError(
                "`dilation` be None, a positive integer, or a sequence of "
                "positive integers with the same length as `window_shape` "
                "({}), got: {}".format(window_shape, dilation)
            )
        if any(
            w * d > s
            for w, d, s in zip(window_shape[::-1], dilation[::-1], arr.shape[::-1])
        ):
            raise ValueError(
                "The dilated window ({}) must fit within the trailing "
                "dimensions of `arr` ({})".format(
                    tuple(w * d for w, d in zip(window_shape, dilation)),
                    arr.shape[-len(window_shape) :],
                )
            )

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    step = np.array(step)  # (Sx, ..., Sz)
    window_shape = np.array(window_shape)  # (Wx, ..., Wz)
    in_shape = np.array(arr.shape[-len(step) :])  # (x, ... , z)
    nbyte = arr.strides[-1]  # size, in bytes, of element in `arr`

    # per-byte strides required to fill a window
    win_stride = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)

    # per-byte strides required to advance the window
    step_stride = tuple(win_stride[-len(step) :] * step)

    # update win_stride to accommodate dilation
    win_stride = np.array(win_stride)
    win_stride[-len(step) :] *= dilation
    win_stride = tuple(win_stride)

    # tuple of bytes to step to traverse corresponding dimensions of view
    # see: 'internal memory layout of an ndarray'
    stride = tuple(int(nbyte * i) for i in step_stride + win_stride)

    # number of window placements along x-dim: X = (x - (Wx - 1)*Dx + 1) // Sx + 1
    out_shape = tuple((in_shape - ((window_shape - 1) * dilation + 1)) // step + 1)

    # ([X, (...), Z], ..., [Wx, (...), Wz])
    out_shape = out_shape + arr.shape[: -len(step)] + tuple(window_shape)
    out_shape = tuple(int(i) for i in out_shape)

    return as_strided(arr, shape=out_shape, strides=stride, writeable=False)
