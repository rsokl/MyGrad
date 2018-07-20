from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
import numpy as np
from numbers import Integral
from mygrad.nnet.layers.utils import sliding_window_view


__all__ = ["conv_nd"]


class ConvND(Operation):
    scalar_only = True

    def __call__(self, x, w, stride, padding=0):
        self.variables = (x, w)
        # x ... data:    (N, C, X0, X1, ...)
        # w ... filters: (F, C, W0, W1, ...)

        x = x.data
        w = w.data

        assert x.ndim > 2
        assert x.ndim == w.ndim
        assert w.shape[1] == x.shape[1], "The channel-depth of the batch and filters must agree"

        num_conv_channels = w.ndim - 2
        x_shape = np.array(x.shape[2:])  # (X0, ...): shape of the channels being convolved over
        w_shape = np.array(w.shape[2:])  # (W0, ...): shape of each conv filter

        padding = np.array((padding,) * num_conv_channels) if isinstance(padding, Integral) else np.array(padding, dtype=int)
        assert len(padding) == num_conv_channels and all(p >= 0 and isinstance(p, Integral) for p in padding)

        stride = np.array((stride,) * num_conv_channels) if isinstance(stride, Integral) else np.asarray(stride, dtype=int)
        assert len(stride) == num_conv_channels and all(s >= 1 and isinstance(s, Integral) for s in stride)

        out_shape = (x_shape + 2 * padding - w_shape) / stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += "Input dimensions: {}\n".format(tuple(x_shape))
            msg += "Stride dimensions: {}\n".format(tuple(stride))
            msg += "Kernel dimensions: {}\n".format(tuple(w_shape))
            msg += "Padding dimensions: {}\n".format(tuple(padding))
            raise ValueError(msg)

        self.padding = padding
        self.stride = stride
        self.dilation = (1,) * num_conv_channels

        # symmetric 0-padding for X0, X1, ... dimensions
        axis_pad = tuple((i, i) for i in (0, 0, *padding))
        x = np.pad(x, axis_pad, mode='constant') if sum(padding) else x

        # (G0, ...) is the tuple of grid-positions for placing each window (not including stride)
        # (N, C, X0, ...) -> (G0, ..., N, C, W0, ...)
        windowed_data = sliding_window_view(x,
                                            window_shape=w_shape,
                                            step=self.stride,
                                            dilation=self.dilation)

        w_conv_channels = list(range(1, num_conv_channels + 2))  # C, W0, ...
        window_conv_channels = [i + 1 + num_conv_channels        # C, W0, ...
                                for i in range(num_conv_channels + 1)]

        # (F, C, W0, ...) ⋆ (G0, ..., N, C, W0, ...) -> (F, G0, ..., N)
        conv_out = np.tensordot(w,
                                windowed_data,
                                axes=[w_conv_channels, window_conv_channels])

        # (F, G0, ..., N) -> (N, F, G0, ...)
        out = np.moveaxis(conv_out, source=-1, destination=0)
        return out if out.flags['C_CONTIGUOUS'] else np.ascontiguousarray(out)

    def backward_var(self, grad, index, **kwargs):
        """ Computes dX, where X is the data batch

            Parameters
            ----------
            grad : numpy.ndarray, shape=(N, F, G0, ...)"""
        x, w = (i.data for i in self.variables)
        num_conv_channels = grad.ndim - 2

        if index == 0:  # backprop through x
            x_shape = x.shape[:2] + tuple(i+2*p for i, p in zip(x.shape[-num_conv_channels:], self.padding))
            dx = np.zeros(x_shape, dtype=x.dtype)  # (N, C, X0, ...)

            # `gp` stores all of the various broadcast multiplications of each grad
            # element against the conv filter.
            # (N, F, G0, ...) -tdot- (F, C, W0, ...) --> (N, G0, ..., C, W0, ...)
            gp = np.tensordot(grad, w, axes=[[1], [0]])
            for ind in np.ndindex(grad.shape[-num_conv_channels:]):
                # ind: (g0, ...) - grid-position of filter placement
                slices = tuple(slice(i * s, i * s + w * d, d) for i, w, s, d in zip(ind, w.shape[2:],
                                                                                    self.stride, self.dilation))
                # Add (grad-element * filter) to each appropriate window position in `dx`
                # dx[N, C, g0*s0 : g0*s0 + w0, (...)] += gp[N, g0, (...), C, W0, (...)]
                dx[(..., *slices)] += gp[(slice(None), *ind, ...)]

            # remove padding from dx
            if sum(self.padding):
                no_pads = tuple(slice(p, -p if p else None) for p in self.padding)
                dx = dx[(..., *no_pads)]
            self.variables[index].backward(dx, **kwargs)

        else:  # backprop through w
            # backprop into f
            # symmetric 0-padding for H, W dimensions
            axis_pad = tuple((i, i) for i in (0, 0, *self.padding))
            x = np.pad(x, axis_pad, mode='constant') if sum(self.padding) else x

            # (G0, ...) is the tuple of grid-indices for placing each window (not including stride)
            # (N, C, X0, ...) -> (G0, ..., N, C, W0, ...)
            windowed_data = sliding_window_view(x,
                                                window_shape=w.shape[2:],
                                                step=self.stride,
                                                dilation=self.dilation)

            # (N, F, G0, ...) -tdot- (G0, ..., N, C, W0, ...) --> (F, C, W0, ...)
            grad_axes = list(range(2, num_conv_channels + 2)) + [0]  # (G0, ..., N)
            window_axes = list(range(num_conv_channels + 1))         # (G0, ..., N)
            df = np.tensordot(grad, windowed_data, axes=[grad_axes, window_axes])
            self.variables[index].backward(df, **kwargs)


def conv_nd(x, filter_bank, stride, padding=0, constant=False):
    """ Use `filter_bank` to perform strided N-dimensional neural network-style
        convolutions (see Notes) over `x`.
                           f(x, w) -> x ⋆ w

                shapes:
                (N, C, X0, ...) ⋆ (F, C, W0, ...) -> (N, F, G0, ...)

        `x` represents a batch of data over which the filters
        are convolved. Specifically, it must be a tensor of shape
        (N, C, X0, ...), where N is the number of samples in the batch,
        C is the channel-depth of each datum, and (X0, ...) are the
        dimensions over which the filters are convolved. Accordingly,
        each filter must have a channel depth of C.

        Thus convolving F filters, each with a shape (C, W0, ...),
        over the data batch will produce a tensor of shape
        (N, F, G0, ...), where (G0, ...) is the shape of the grid
        commensurate with the filter placements

        Parameters
        ----------
        x : Union[Tensor, array_like], shape=(N, C, X0, ...)
            The data batch to be convolved over.

        filter_bank : Union[Tensor, array_like], shape=(F, C, W0, ...)
            The filters used to perform the convolutions.

        stride : Union[int, Tuple[int]]
            The step-size with which each filter is placed along the
            H and W axes during the convolution. The tuple indicates
            (stride-0, ...). If a single integer is provided, this
            stride is used for all convolved dimensions

        padding : Union[int, Tuple[int]]
            The number of zeros to be padded to both ends of
            each convolved dimension, respectively. If a single
            integer is provided, this padding is used for all of
            the convolved axes

        constant : bool, optional (default=False)
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
           `tensor.backward()` for the computational graph. This is standard for a
           neural network, which terminates in a scalar loss."""
    return Tensor._op(ConvND, x, filter_bank, op_args=(stride, padding), constant=constant)
