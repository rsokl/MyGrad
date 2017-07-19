from ..operations.operation_base import Operation
from ..tensor_base import Tensor
from .utils import im2col, col2im
from numbers import Integral
import numpy as np

__all__ = ["dense", "conv2d", "max_pool"]

class Dense(Operation):
    scalar_only = True

    def __call__(self, a, b):
        assert a.ndim == 2 and b.ndim == 2
        self.a = a
        self.b = b
        return np.dot(a.data, b.data)

    def backward_a(self, grad):
        self.a.backward(np.dot(grad, self.b.data.T))

    def backward_b(self, grad):
        self.b.backward(np.dot(self.a.data.T, grad))


def dense(x, w):
    """ Perform a dense-layer pass (i.e. matrix multiplication) of a (N, D)-shape
        tensor with a (D, M)-shape tensor.

        Parameters
        ----------
        x : Union[mygrad.Tensor, array_like], shape=(N, D)

        w : Union[mygrad.Tensor, array_like], shape=(D, M)

        Returns
        -------
        Tensor, shape=(N, M)
            The result of the matrix multiplication of `x` with `w`

        Notes
        -----
        This is a "scalar-only" operation, meaning that back propagation through
        this layer assumes that a scalar (i.e. a 0-dimensional tensor) will invoke
        `tensor.backward()` for the computational graph. This is standard for a
        neural network, which terminates in a scalar loss."""
    return Tensor._op(Dense, x, w)


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

    def backward(self, grad):
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
    """ Use `filter_bank` to perform strided 2D convolutions (see Notes) over `x`.

        `x` represents a batch of data over which the filters are convolved. Specifically,
        it must be a tensor of shape (N, C, H, W), where N is the number of samples in the batch,
        C is the channel-depth of each datum, and H & W are the dimensions over which the filters
        are convolved. Accordingly, each filter must have a channel depth of C.

        Thus convolving F filters over the data batch will produce a tensor of shape (N, F, H', W').

        Parameters
        ----------
        x : Union[Tensor, array_like], shape=(N, C, H, W)
            The data batch to be convolved over.

        filter_bank : Union[Tensor, array_like], shape=(F, C, Hf, Wf)
            The filters used to perform the convolutions.

        stride : Union[int, Tuple[int, int]]
            The step-size with which each filter is placed along the H and W axes
            during the convolution. The tuple indicates (stride-H, stride-W). If a
            single integer is provided, this stride is used for both axes.

        padding : Tuple[int, int]
            The number of zeros to be padded to either end of the H-dimension
            and the W-dimension, respectively, for each datum in the batch. If
            a single integer is provided, this padding is used for both axes

        memory_constrained : Bool, optional (default=False)
            By default, a 'stretched' version of the data batch (see Notes) is cached
            after the convolution is performed, for use when performing `backward`.

            Setting this to False will forego caching, at the expense of some computation
            time during `backward`.

        Returns
        -------
        Tensor, shape=(N, F, H', W')
            The result of each filter being convolved over each datum in the batch.

        Notes
        -----
         - The filters are *not* flipped by this operation, meaning that an auto-correlation
           is being performed rather than a true convolution.

         - The present implementation of `conv2d` relies on performing a matrix multiplication
           in lieu of a true auto-correlation. This comes at the cost of a significant memory
           footprint, especially if a computational graph contains multiple convolutions. Consider
           setting `memory_constrained` to True in these circumstances.

         - Only 'valid' filter placements are permitted - where the filters overlap completely
           with the (padded) data.

         - This is a "scalar-only" operation, meaning that back propagation through
           this layer assumes that a scalar (i.e. a 0-dimensional tensor) will invoke
           `tensor.backward()` for the computational graph. This is standard for a
           neural network, which terminates in a scalar loss."""
    return Tensor._op(Conv2D, x, filter_bank, op_args=(stride, padding, memory_constrained))


class MaxPool(Operation):
    def __call__(self, x, pool, stride):
        """ Perform max-pooling over the last two dimensions of a data batch.

            Parameters
            ----------
            x : Tensor, shape=(N, C, H, W)
                The data batch; to be pooled along the H and W axes

            pool : Union[int, Tuple[int, int]]
                The extent of the pooling window along the H and W axes, respectively. If
                a single value is provided, it is used for both axes.

            stride : Union[int, Tuple[int, int]]
                The spacing used to place the pooling window, along the H and W axes, respectively.
                If a single value is provided, it is used for both axes.

            Returns
            -------
            numpy.ndarray, shape=(N, C, H', W')
                The pooled data batch.

            Notes
            -----
            Only 'valid' placements of the pooling window are permitted - the pooling
            window cannot extend passed the "boundaries" of the data in the H and W
            dimensions.
            """
        self.a = x  # data: (N, C, H, W)
        x = x.data

        assert x.ndim == 4, "The data batch must have the shape (N, C, H, W)"
        stride = np.array((stride, stride)) if isinstance(stride, Integral) else np.asarray(stride, dtype=int)
        assert len(stride) == 2 and np.all(stride >= 1) and np.issubdtype(stride.dtype, np.int)

        pool = np.array((pool, pool)) if isinstance(pool, Integral) else np.asarray(pool, dtype=int)
        assert len(pool) == 2 and np.all(pool >= 1) and np.issubdtype(pool.dtype, np.int)
        pool = (1, 1) + tuple(pool)

        x_shape = np.array(x.shape[2:])
        w_shape = np.array(pool[2:])

        out_shape = (x_shape - w_shape) / stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += "Input dimensions: {}\n".format(tuple(x_shape))
            msg += "Stride dimensions: {}\n".format(tuple(stride))
            msg += "Pooling dimensions: {}\n".format(tuple(w_shape))
            raise ValueError(msg)

        n, c, ho, wo = (x.shape[0], x.shape[1]) + tuple(out_shape.astype(int))

        x_reshaped = x.reshape(x.shape[0] * x.shape[1], 1, *x.shape[2:])
        x_col = im2col(x_reshaped, pool, (n, c, ho, wo), [0, 0], stride)

        max_idx = np.argmax(x_col, axis=0)
        out = x_col[max_idx, range(max_idx.size)]

        self.cache = max_idx
        self.pool = pool
        self.stride = stride
        self.col_shape = x_col.shape
        return out.reshape(ho, wo, n, c).transpose(2, 3, 0, 1)

    def backward_a(self, grad):
        max_id = self.cache

        dx_col = np.zeros(self.col_shape, dtype=self.a.dtype)
        dx_col[max_id, range(grad.size)] = grad.transpose(2, 3, 0, 1).ravel()
        n, c, h, w = self.a.shape
        dx = col2im(dx_col, (n * c, 1, h, w), self.pool, grad.shape, [0, 0], self.stride)
        self.a.backward(dx.reshape(self.a.shape))


def max_pool(x, pool, stride):
    """ Perform max-pooling over the last two dimensions of a data batch.

        Parameters
        ----------
        x : Tensor, shape=(N, C, H, W)
            The data batch; to be pooled along the H and W axes

        pool : Union[int, Tuple[int, int]]
            The extent of the pooling window along the H and W axes, respectively. If
            a single value is provided, it is used for both axes.

        stride : Union[int, Tuple[int, int]]
            The spacing used to place the pooling window, along the H and W axes, respectively.
            If a single value is provided, it is used for both axes.

        Returns
        -------
        Tensor, shape=(N, C, H', W')
            The pooled data batch.

        Notes
        -----
        Only 'valid' placements of the pooling window are permitted - the pooling
        window may not extend passed the "boundaries" of the data in the H and W
        dimensions.

        This is a memory-intensive implementation of max pooling. """
    return Tensor._op(MaxPool, x, op_args=(pool, stride))

