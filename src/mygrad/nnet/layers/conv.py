from numbers import Integral
from typing import Optional, Tuple, Union

import numpy as np

from mygrad.nnet.layers.utils import sliding_window_view
from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike

__all__ = ["conv_nd"]


class ConvND(Operation):
    def __call__(self, x, w, *, stride, padding=0, dilation=1):
        self.variables = (x, w)
        # x ... data:    (N, C, X0, X1, ...)
        # w ... filters: (F, C, W0, W1, ...)

        x = x.data
        w = w.data

        assert x.ndim > 2
        assert x.ndim == w.ndim
        assert (
            w.shape[1] == x.shape[1]
        ), "The channel-depth of the batch and filters must agree"

        num_conv_channels = w.ndim - 2
        x_shape = np.array(
            x.shape[2:]
        )  # (X0, ...): shape of the channels being convolved over
        w_shape = np.array(w.shape[2:])  # (W0, ...): shape of each conv filter

        dilation = (
            np.array((dilation,) * num_conv_channels)
            if isinstance(dilation, Integral)
            else np.array(dilation, dtype=int)
        )

        assert len(dilation) == num_conv_channels and all(
            d >= 1 and isinstance(d, Integral) for d in dilation
        )

        padding = (
            np.array((padding,) * num_conv_channels)
            if isinstance(padding, Integral)
            else np.array(padding, dtype=int)
        )
        assert len(padding) == num_conv_channels and all(
            p >= 0 and isinstance(p, Integral) for p in padding
        )

        stride = (
            np.array((stride,) * num_conv_channels)
            if isinstance(stride, Integral)
            else np.asarray(stride, dtype=int)
        )
        assert len(stride) == num_conv_channels and all(
            s >= 1 and isinstance(s, Integral) for s in stride
        )

        out_shape = (
            x_shape + 2 * padding - ((w_shape - 1) * dilation + 1)
        ) / stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += f"Input dimensions: {tuple(x_shape)}\n"
            msg += f"Stride dimensions: {tuple(stride)}\n"
            msg += f"Kernel dimensions: {tuple(w_shape)}\n"
            msg += f"Padding dimensions: {tuple(padding)}\n"
            msg += f"Dilation dimensions: {tuple(dilation)}\n"
            raise ValueError(msg)

        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        # symmetric 0-padding for X0, X1, ... dimensions
        axis_pad = tuple((i, i) for i in (0, 0, *padding))
        x = np.pad(x, axis_pad, mode="constant") if sum(padding) else x

        # (G0, ...) is the tuple of grid-positions for placing each window (not including stride)
        # (N, C, X0, ...) -> (G0, ..., N, C, W0, ...)
        windowed_data = sliding_window_view(
            x, window_shape=w_shape, step=self.stride, dilation=self.dilation
        )

        w_conv_channels = list(range(1, num_conv_channels + 2))  # C, W0, ...
        window_conv_channels = [
            i + 1 + num_conv_channels  # C, W0, ...
            for i in range(num_conv_channels + 1)
        ]

        # (F, C, W0, ...) ⋆ (G0, ..., N, C, W0, ...) -> (F, G0, ..., N)
        conv_out = np.tensordot(
            w, windowed_data, axes=[w_conv_channels, window_conv_channels]
        )

        # (F, G0, ..., N) -> (N, F, G0, ...)
        out = np.moveaxis(conv_out, source=-1, destination=0)
        return out if out.flags["C_CONTIGUOUS"] else np.ascontiguousarray(out)

    def backward_var(self, grad, index, **kwargs):
        """Computes dX, where X is the data batch

        Parameters
        ----------
        grad : numpy.ndarray, shape=(N, F, G0, ...)"""
        x, w = (i.data for i in self.variables)
        num_conv_channels = grad.ndim - 2

        if index == 0:  # backprop through x
            x_shape = x.shape[:2] + tuple(
                i + 2 * p for i, p in zip(x.shape[-num_conv_channels:], self.padding)
            )
            dx = np.zeros(x_shape, dtype=x.dtype)  # (N, C, X0, ...)

            # `gp` stores all of the various broadcast multiplications of each grad
            # element against the conv filter.
            # (N, F, G0, ...) -tdot- (F, C, W0, ...) --> (N, G0, ..., C, W0, ...)
            gp = np.tensordot(grad, w, axes=[[1], [0]])
            for ind in np.ndindex(grad.shape[-num_conv_channels:]):
                # ind: (g0, ...) - grid-position of filter placement
                slices = tuple(
                    slice(i * s, i * s + w * d, d)
                    for i, w, s, d in zip(ind, w.shape[2:], self.stride, self.dilation)
                )
                # Add (grad-element * filter) to each appropriate window position in `dx`
                # dx[N, C, g0*s0 : g0*s0 + w0*d0 : d0, (...)] += gp[N, g0, (...), C, W0, (...)]
                dx[(..., *slices)] += gp[(slice(None), *ind, ...)]

            # remove padding from dx
            if sum(self.padding):
                no_pads = tuple(slice(p, -p if p else None) for p in self.padding)
                dx = dx[(..., *no_pads)]
            return dx

        else:  # backprop through w
            # backprop into f
            # symmetric 0-padding for H, W dimensions
            axis_pad = tuple((i, i) for i in (0, 0, *self.padding))
            x = np.pad(x, axis_pad, mode="constant") if sum(self.padding) else x

            # (G0, ...) is the tuple of grid-indices for placing each window (not including stride)
            # (N, C, X0, ...) -> (G0, ..., N, C, W0, ...)
            windowed_data = sliding_window_view(
                x, window_shape=w.shape[2:], step=self.stride, dilation=self.dilation
            )

            # (N, F, G0, ...) -tdot- (G0, ..., N, C, W0, ...) --> (F, C, W0, ...)
            grad_axes = list(range(2, num_conv_channels + 2)) + [0]  # (G0, ..., N)
            window_axes = list(range(num_conv_channels + 1))  # (G0, ..., N)
            return np.tensordot(grad, windowed_data, axes=[grad_axes, window_axes])


def conv_nd(
    x: ArrayLike,
    filter_bank: ArrayLike,
    *,
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    constant: Optional[bool] = None,
) -> Tensor:
    """Use `filter_bank` to perform strided N-dimensional neural network-style
    convolutions (see Notes) over `x`.::

            f(x, w) -> x ⋆ w

            shapes:
            (N, C, X0, ...) ⋆ (F, C, W0, ...) -> (N, F, G0, ...)

    ``x`` represents a batch of data over which the filters
    are convolved. Specifically, it must be a tensor of shape
    :math:`(N, C, X_0, ...)`, where :math:`N` is the number of samples in the batch,
    C is the channel-depth of each datum, and :math:`(X_0, ...)` are the
    dimensions over which the filters are convolved. Accordingly,
    each filter must have a channel depth of :math:`C`.

    Thus convolving :math:`F` filters, each with a shape :math:`(C, W_0, ...)`,
    over the data batch will produce a tensor of shape
    :math:`(N, F, G_0, ...)`, where :math:`(G_0, ...)` is the shape of the grid
    commensurate with the filter placements

    Parameters
    ----------
    x : Union[Tensor, array_like], shape=(N, C, Xo, ...)
        The data batch to be convolved over.

    filter_bank : Union[Tensor, array_like], shape=(F, C, Wo, ...)
        The filters used to perform the convolutions.

    stride : Union[int, Tuple[int, ...]]
        (keyword-only argument) The step-size with which each
        filter is placed along the H and W axes during the
        convolution. The tuple indicates (stride-0, ...). If a
        single integer is provided, this stride is used for all
        convolved dimensions

    padding : Union[int, Tuple[int, ...]]
        (keyword-only argument) The number of zeros to be padded
        to both ends of each convolved dimension, respectively.
        If a single integer is provided, this padding is used for
        all of the convolved axes

    dilation : Union[int, Tuple[int, ...]], optional (default=1)
        (keyword-only argument) The spacing used when placing kernel
        elements along the data. E.g. for a 1D convolution the ith
        placement of the kernel multiplied  against the dilated-window:
        ``x[:, :, i*s:(i*s + w*d):d]``, where ``s`` is
        the stride, ``w`` is the kernel-size, and ``d`` is the dilation factor.

        If a single integer is provided, that dilation value is used for all
        of the convolved axes

    constant : Optional[None]
        If True, the resulting Tensor is a constant.

    Returns
    -------
    Tensor, shape=(N, F, G0, ...)
        The result of each filter being convolved over each datum in
        the batch.

    Notes
    -----
     - The filters are *not* flipped by this operation, meaning that
       an auto-correlation is being performed rather than a true convolution.

     - Only 'valid' filter placements are permitted - where the filters overlap
       completely with the (padded) data.

     - This is a "scalar-only" operation, meaning that back propagation through
       this layer assumes that a scalar (i.e. a 0-dimensional tensor) will invoke
       ``tensor.backward()`` for the computational graph. This is standard for a
       neural network, which terminates in a scalar loss.

    Examples
    --------
    Here we perform a 1D convolution of a constant-valued kernel, ``k``, with a
    'square-wave' signal, ``x``, using stride-1. Note that because we are constrained
    to doing deep learning-style convolutions, that we prepend the dimensions
    :math:`(N=1, C=1)` to ``x``, and :math:`(F=1, C=1)` and to ``k``. That is,
    we are performing a convolution on one, single-channeled signal using
    one kernel.

    See that this convolution produces the expected triangle-shaped
    response. The shape of the resulting tensor is :math:`(N=1, F=1, G_0=12)`.
    That is, the length-5 kernel can be placed in 12 valid positions, using a
    stride of 1.

    >>> import mygrad as mg
    >>> from mygrad.nnet import conv_nd
    >>> x = mg.zeros((1, 1, 16))  # a square-wave signal
    >>> x[..., 5:11] = 1
    >>> k = mg.ones((1, 1, 5))    # a constant-valued kernel
    >>> conv_nd(x, k, stride=1)   # performing a stride-1, 1D convolution
    Tensor([[[0., 1., 2., 3., 4., 5., 5., 4., 3., 2., 1., 0.]]], dtype=float32)

    Back-propagating through the (summed) convolution:

    >>> conv_nd(x, k, stride=1).sum().backward()  # sum to a scalar to perform back-prop
    >>> x.grad  # d(summed_conv)/dx
    array([[[1., 2., 3., 4., 5., 5., 5., 5., 5., 5., 5., 5., 4., 3., 2., 1.]]],
          dtype=float32)
    >>> k.grad  # d(summed_conv)/dk
    array([[[6., 6., 6., 6., 6.]]])

    Now, let's demonstrate a more typical usage for ``conv_nd`` in the context of
    neural networks. ``x`` will represent 10, 32x32 RGB images, and we will use
    5 distinct 2x2 kernels to convolve over each of these images . Note that
    each kernel must possess 3-channel - one for each RGB channel.

    That is, we will be performing NxF channel-wise 2D convolutions. Supposing
    that we don't want the kernel placements to overlap, we can use a stride of 2. In
    total, this will produce a shape-:math:`(N=10, F=5, G_0=16, G_1=16)` tensor as a
    result.

    >>> import numpy as np
    >>> x = mg.Tensor(np.random.rand(10, 3, 32, 32))  # creating 10 random 32x32 RGB images
    >>> k = mg.Tensor(np.random.rand(5, 3, 2, 2))     # creating 5 random 3-channel 2x2 kernels

    Given the shapes of ``x`` and ``k``, ``conv_nd`` automatically executes a 2D convolution:

    >>> conv_nd(x, k, stride=2).shape
    (10, 5, 16, 16)

    Extrapolating further, ``conv_nd`` is capable of performing ND convolutions!
    """
    if x.ndim < 3:
        raise ValueError(
            f"`x` must possess at least three " f"dimensions, got {x.ndim} dimensions"
        )

    if x.ndim != filter_bank.ndim:
        raise ValueError(
            f"`x` ({x.ndim}-dimensions) must have the same dimensionality as "
            f"`filter_bank` ({filter_bank.ndim}-dimensions)"
        )

    if filter_bank.shape[1] != x.shape[1]:
        raise ValueError(
            f"`x.shape[1]` ({x.shape[1]}) must match `filter_bank.shape[1]` ({filter_bank.shape[1]})"
        )

    return Tensor._op(
        ConvND,
        x,
        filter_bank,
        op_kwargs=dict(stride=stride, padding=padding, dilation=dilation),
        constant=constant,
    )
