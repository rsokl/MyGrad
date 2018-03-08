from ...operations.operation_base import Operation
from ...tensor_base import Tensor
import numpy as np
from numbers import Integral
from mygrad.nnet.layers.utils import sliding_window_view


class Conv2D(Operation):
    def __call__(self, x, w, stride, padding=(0, 0)):
        self.a = x  # data: (N, C, H, W)
        self.b = w  # filters: (F, C, Hf, Wf)

        x = x.data
        w = w.data

        assert x.ndim == 4, "The data batch must have the shape (N, C, H, W)"
        assert w.ndim == 4, "The filter bank must have the shape (F, C, Hf, Wf)"
        assert w.shape[1] == x.shape[1], "The channel-depth of the batch and filters must agree"

        x_shape = np.array(x.shape[2:])
        w_shape = np.array(w.shape[2:])

        padding = np.array((padding, padding)) if isinstance(padding, Integral) else np.array(padding, dtype=int)
        assert len(padding) == 2 and np.all(padding >= 0) and np.issubdtype(padding.dtype, np.int)

        stride = np.array((stride, stride)) if isinstance(stride, Integral) else np.asarray(stride, dtype=int)
        assert len(stride) == 2 and np.all(stride >= 1) and np.issubdtype(stride.dtype, np.int)

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
        self.dilation = (1, 1)

        # symmetric 0-padding for H, W dimensions
        axis_pad = tuple((i, i) for i in (0, 0, *padding))
        x = np.pad(x, axis_pad, mode='constant') if sum(padding) else x

        windowed_data = sliding_window_view(x,
                                            window_shape=w.shape[-2:],
                                            step=self.stride,
                                            dilation=self.dilation)

        conv_out = np.tensordot(w,
                                windowed_data,
                                axes=[[1, 2, 3], [3, 4, 5]])
        # (F, H', W', N) -> (N, F, H', W')
        return conv_out.transpose([3, 0, 1, 2])

    def backward_a(self, grad):
        """ Computes dX, where X is the data batch"""
        x = self.a.data
        w = self.b.data

        x_shape = x.shape[:-2] + tuple(i+2*p for i, p in zip(x.shape[-2:], self.padding))
        dx = np.zeros(x_shape, dtype=x.dtype)  # (N, C, H + 2*ph, W + 2*pw)

        # `gp` stores all of the various broadcast multiplications of each grad
        # element against the conv filter.
        # (N, F, H', W') -tdot- (F, C, Hf, Wf) --> (N, H', W', C, Hf, Wf)
        gp = np.tensordot(grad, w, axes=[[1], [0]])
        for ind in np.ndindex(grad.shape[-len(self.stride):]):
            # ind: (h', w') - grid-position of filter placement
            slices = tuple(slice(i * s, i * s + w * d, d) for i, w, s, d in zip(ind, w.shape[-2:],
                                                                                self.stride, self.dilation))
            # Add (grad-element * filter) to each appropriate window position in `dx`
            # dx[N, C, h'*sh : h'*sh + Wh, w'*sw : w'*sh + Wh] += gp[N, h', w', C, Hf, Wf]
            dx[(..., *slices)] += gp[(slice(None), *ind, ...)]

        # remove padding from dx
        if sum(self.padding):
            no_pads = tuple(slice(p, -p if p else None) for p in self.padding)
            dx = dx[(..., *no_pads)]
        self.a.backward(dx)

    def backward_b(self, grad):
        """ Computes dW, where W are the conv filters"""
        x = self.a.data
        w = self.b.data
        # backprop into f
        # symmetric 0-padding for H, W dimensions
        axis_pad = tuple((i, i) for i in (0, 0, *self.padding))
        x = np.pad(x, axis_pad, mode='constant') if sum(self.padding) else x

        windowed_data = sliding_window_view(x,
                                            window_shape=w.shape[-2:],
                                            step=self.stride,
                                            dilation=self.dilation)

        # (N, F, H', W') -tdot- (H', W', N, C, Hf, Wf) --> (F, C, Hf, Wf)
        df = np.tensordot(grad, windowed_data, axes=[[2, 3, 0], [0, 1, 2]])
        self.b.backward(df)


def conv2d(x, filter_bank, stride, padding=(0, 0), memory_constrained=False):
    """ Use `filter_bank` to perform strided 2D convolutions
       (see Notes) over `x`.

        `x` represents a batch of data over which the filters
        are convolved. Specifically, it must be a tensor of shape
        (N, C, H, W), where N is the number of samples in the batch,
        C is the channel-depth of each datum, and H & W are the
        dimensions over which the filters are convolved. Accordingly,
        each filter must have a channel depth of C.

        Thus convolving F filters over the data batch will produce a
        tensor of shape (N, F, H', W').

        Parameters
        ----------
        x : Union[Tensor, array_like], shape=(N, C, H, W)
            The data batch to be convolved over.

        filter_bank : Union[Tensor, array_like], shape=(F, C, Hf, Wf)
            The filters used to perform the convolutions.

        stride : Union[int, Tuple[int, int]]
            The step-size with which each filter is placed along the
            H and W axes during the convolution. The tuple indicates
            (stride-H, stride-W). If a single integer is provided, this
            stride is used for both axes.

        padding : Tuple[int, int]
            The number of zeros to be padded to either end of
            the H-dimension and the W-dimension, respectively,
            for each datum in the batch. If a single integer is
            provided, this padding is used for both axes

        memory_constrained : Bool, optional (default=False)
            By default, a 'stretched' version of the data batch
            (see Notes) is cached after the convolution is performed,
            for use when performing `backward`.

            Setting this to False will forego caching, at the expense
            of some computation time during `backward`.

        Returns
        -------
        Tensor, shape=(N, F, H', W')
            The result of each filter being convolved over each datum in
            the batch.

        Notes
        -----
         - The filters are *not* flipped by this operation, meaning that
           an auto-correlation is being performed rather than a true convolution.

         - The present implementation of `conv2d` relies on performing a
           matrix multiplication in lieu of a true auto-correlation. This
           comes at the cost of a significant memory footprint, especially
           if a computational graph contains multiple convolutions. Consider
           setting `memory_constrained` to True in these circumstances.

         - Only 'valid' filter placements are permitted - where the filters overlap
           completely with the (padded) data.

         - This is a "scalar-only" operation, meaning that back propagation through
           this layer assumes that a scalar (i.e. a 0-dimensional tensor) will invoke
           `tensor.backward()` for the computational graph. This is standard for a
           neural network, which terminates in a scalar loss."""
    return Tensor._op(Conv2D, x, filter_bank, op_args=(stride, padding))