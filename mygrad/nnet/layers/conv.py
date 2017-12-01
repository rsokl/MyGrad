from ...operations.operation_base import Operation
from ...tensor_base import Tensor
import numpy as np
from numbers import Integral
from mygrad.nnet.layers.utils import im2col, col2im

class Conv2D(Operation):
    """ This actually performs an auto-correlation, and not a convolution."""

    def __call__(self, x, w, stride, padding, memory_constrained=False):
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

        out_shape = (x.shape[0], w.shape[0]) + tuple(out_shape.astype(int))  # (N, F, H', W')

        # perform 2D convolution as matrix multiplication
        x_col = im2col(x, w.shape, out_shape, padding, stride)  # (Hf * Wf * C, N')
        w_col = w.reshape(w.shape[0], -1)  # (F, Hf * Wf * C)

        # (F, N') -> (N, F, H', W')
        out = (w_col.dot(x_col)).reshape(out_shape[1:] + out_shape[0:1]).transpose(3, 0, 1, 2)

        self.cache = x_col if not (self.b.constant or memory_constrained) else None
        self.out_shape = out_shape
        self.padding = padding
        self.stride = stride
        return out

    def backward(self, grad, **kwargs):
        if self.a.constant and self.b.constant:
            return None

        grad = grad.transpose(1, 2, 3, 0).reshape(self.b.shape[0], -1)
        super(Conv2D, self).backward(grad)

    def backward_a(self, grad):
        """ Computes dX, where X is the data batch"""
        x = self.a.data
        w = self.b.data

        w_reshape = w.reshape(w.shape[0], -1).T
        dx = col2im(w_reshape.dot(grad), x.shape, w.shape, self.out_shape, self.padding, self.stride)
        self.a.backward(dx)

    def backward_b(self, grad):
        x = self.a.data
        w = self.b.data
        x_col = self.cache if self.cache is not None else im2col(x, w.shape, self.out_shape,
                                                                 self.padding, self.stride)
        self.b.backward(grad.dot(x_col.T).reshape(w.shape))


def conv2d(x, filter_bank, stride, padding, memory_constrained=False):
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
    return Tensor._op(Conv2D, x, filter_bank, op_args=(stride, padding, memory_constrained))