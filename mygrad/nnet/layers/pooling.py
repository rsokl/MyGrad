from ...operations.operation_base import Operation
from ...tensor_base import Tensor
import numpy as np
from numbers import Integral
from mygrad.nnet.layers.utils import im2col, col2im


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

