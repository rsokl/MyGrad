from ...operations.operation_base import Operation
from ...tensor_base import Tensor
import numpy as np
from numbers import Integral
from mygrad.nnet.layers.utils import im2col, col2im, sliding_window_view


class MaxPoolND(Operation):
    def __call__(self, x, pool, stride):
        """ Perform max-pooling over the last N dimensions of a data batch.

            Parameters
            ----------
            x : mygrad.Tensor, shape=([(N0, ...), C0, ...])
                The data batch; to be pooled along the H and W axes

            pool : Tuple[Integral, ...]
                (P0, ...)

                The extent of the pooling window along the (C0, ...) axes, respectively. The
                length of `pool` determines N - the number of trailing dimensions to pool over.

            stride : Union[Integral, Tuple[Integral, ...]]
                The spacing used to place the pooling window, along (P0, ...) axes, respectively.
                If a single value is provided, it is used for all N pooling axes.

            Returns
            -------
            numpy.ndarray, shape=((N0, ...), C0', ...)
                The pooled data batch.

            Notes
            -----
            Only 'valid' placements of the pooling window are permitted - the pooling
            window cannot extend passed the "boundaries" of the data in the H and W
            dimensions.
            """
        self.a = x  # data: ((N0, ...), C0, ...) # TODO: copy data!!
        x = x.data

        pool = np.asarray(pool, dtype=int)
        assert all(i > -1 for i in pool)  # TODO: Check if 0-pool dim is valid..
        assert x.ndim >= len(pool), "The number of pooled dimensions cannot exceed the dimensionality of the data."
        
        stride = np.array([stride]*len(pool)) if isinstance(stride, Integral) else np.asarray(stride, dtype=int)
        assert len(stride) == len(pool) and np.all(stride >= 1) #and np.issubdtype(stride.dtype, int)

        self.pool = pool
        self.stride = stride

        num_pool = len(pool)
        num_no_pool = x.ndim - num_pool

        x_shape = np.array(x.shape[num_no_pool:])
        w_shape = pool

        out_shape = (x_shape - w_shape) / stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += "Input dimensions: {}\n".format(tuple(x_shape))
            msg += "Stride dimensions: {}\n".format(tuple(stride))
            msg += "Pooling dimensions: {}\n".format(tuple(w_shape))
            raise ValueError(msg)

        pool_axes = tuple(-(i + 1) for i in range(num_pool))
        
        # (G0, ...) is the tuple of grid-positions for placing each window (not including stride)
        # sliding_window_view(x): ((N0, ...), C0, ...) -> (G0, ..., (N0, ...), P0, ...)
        # max-pool: (G0, ..., (N0, ...), P0, ...) -> (G0, ..., (N0, ...))
        maxed = sliding_window_view(x, self.pool, self.stride).max(axis=pool_axes)
        axes = tuple(range(maxed.ndim))

        # (G0, ..., (N0, ...)) -> ((N0, ...), G0, ...)
        return maxed.transpose(axes[-num_no_pool:] + axes[:-num_no_pool])

    def backward_a(self, grad):
        """ Parameters
            ----------
            grad : numpy.ndarray, shape=((N0, ...), G0, ...)"""
        
        x = self.a.data
        num_pool = len(self.pool)

        sl = sliding_window_view(x, self.pool, self.stride)
        grid_shape = sl.shape
        maxed = sl.reshape(*sl.shape[:-num_pool], -1).argmax(-1)
        axes = tuple(range(maxed.ndim))
        maxed = maxed.transpose(axes[num_pool:] + axes[:num_pool])  # ((N0, ...), G0, ...)

        row_major_offset = tuple(np.cumprod(x.shape[-num_pool:][:0:-1])[::-1]) + (1,)

        # i*W + j
        in_window_offset = sum(
            ind * off for ind, off in zip(np.unravel_index(maxed, self.pool),
                                          row_major_offset))

        # ih*stride[0]*W + jh*stride[1]
        window_offset = sum(ind * s * off for ind, s, off in zip(np.indices(grid_shape[:num_pool]),
                                                                 self.stride,
                                                                 row_major_offset))

        # ((N0, ...) G0*...)
        flat_grid_shape = (*maxed.shape[:-num_pool], np.prod(maxed.shape[-num_pool:]))
        index = np.indices(flat_grid_shape)
        index[-1] = (in_window_offset + window_offset).reshape(*flat_grid_shape[:-1], -1)

        dx = np.zeros(x.shape[:-num_pool] + (np.prod(x.shape[-num_pool:]),))
        np.add.at(dx, tuple(index), grad.reshape(*x.shape[:-num_pool], -1))
        self.a.backward(dx.reshape(x.shape))


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
    if isinstance(pool, Integral): pool = (pool, pool)
    return Tensor._op(MaxPoolND, x, op_args=(pool, stride))


class MaxPoolOld(Operation):
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


def max_poolold(x, pool, stride):
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
    return Tensor._op(MaxPoolOld, x, op_args=(pool, stride))



class MaxPoolOld(Operation):
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


def max_poolold(x, pool, stride):
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
    return Tensor._op(MaxPoolOld, x, op_args=(pool, stride))


class MaxPoolOld(Operation):
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


def max_poolold(x, pool, stride):
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
    return Tensor._op(MaxPoolOld, x, op_args=(pool, stride))

