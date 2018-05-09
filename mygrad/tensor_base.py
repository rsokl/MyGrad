from mygrad.math.arithmetic.ops import *
from mygrad.math.sequential.ops import *
from mygrad.tensor_manip.transpose_like.ops import Tensor_Transpose_Property, Transpose
from mygrad.tensor_manip.array_shape.ops import Reshape
from mygrad.tensor_core_ops.indexing import GetItem, SetItem
from mygrad.operation_base import Operation, BroadcastableOp
from mygrad._utils import reduce_broadcast

import numpy as np


__all__ = ['Tensor']


class Tensor:
    """ A numpy.array-like object capable of serving as a node in a computational graph that
        supports back-propagation of derivatives via the chain rule."""
    __array_priority__ = 15.0

    def __init__(self, x, *, constant=False, _scalar_only=False, _creator=None):
        """ Parameters
            ----------
            x : array_like
                Input data, in any form that can be converted to an array.  This
                includes numbers, seqeunces, nested sequences, and numpy-ndarrays.

            Keyword-Only Arguments
            ----------------------
            constant : bool, optional (default=False)
                If True, this node is treated as a constant, and thus does not facilitate
                back propagation; `self.grad` will always return `None`.

            _scalar_only : bool, optional (default=False)
                Signals that self.backward() can only be invoked if self.ndim == 0.

            _creator: Optional[mygrad.Operation]
                The operation-instance whose forward pass produced `self`.
            """
        assert isinstance(constant, bool)
        self._scalar_only = _scalar_only
        self._creator = _creator

        if isinstance(x, Tensor):
            self.data = x.data
        else:
            self.data = np.asarray(x)
            self._check_valid_dtype(self.data.dtype)

        self.grad = None
        self._constant = constant

        # used for setitem
        self._ops = []  # Operation instances that utilized self an input tensor

    @staticmethod
    def _check_valid_dtype(dtype):
        if not np.issubdtype(dtype, np.number):
            raise TypeError("Tensor data must be a numeric type")

    @classmethod
    def _op(cls, Op, *input_vars, op_args=None, op_kwargs=None, constant=False):
        """ Wraps operations performed between tensors: f(a, b, ...).

            Parameters
            ----------
            Op : Operation
                Operation-class, used to perform forward-pass on `input_vars`.

            input_vars : Sequence[Union[Number, numpy.ndarray]]
                An arbitrary number of tensor-like objects, which are used as the input
                tensors to the forward-pass of the operation.

            op_args : Optional[Tuple[Any]]
                Arbitrary positional arguments passed to the operation's forward pass.

            op_kwargs : Optional[Dict[str, Any]]
                Arbitrary keyword arguments passed to the operation's forward pass.

            constant : bool, optional (default=False)
                If True, the resulting Tensor is a constant.

            Returns
            -------
            Tensor
                The tensor-result of the operation's forward-pass."""

        if op_args is None:
            op_args = tuple()

        if op_kwargs is None:
            op_kwargs = dict()

        tensor_vars = []
        for var in input_vars:
            if not isinstance(var, cls):
                var = cls(var, constant=True)
            tensor_vars.append(var)

        f = Op()
        op_out = f(*tensor_vars, *op_args, **op_kwargs)

        # check if broadcasting occurred
        if isinstance(f, BroadcastableOp):
            f.scalar_only = any(op_out.shape != i.shape for i in tensor_vars if not i.constant)

        is_const = constant or all(var.constant for var in tensor_vars)

        if not is_const:
            # record that a variable participated in that op
            for var in tensor_vars:
                if not var.constant:
                    var._ops.append(f)

        scalar_only = f.scalar_only and not is_const
        for var in tensor_vars:
            scalar_only = scalar_only or (var.scalar_only and not var.constant)

        return cls(op_out, constant=is_const, _creator=f, _scalar_only=scalar_only)

    def backward(self, grad=None, *, _broadcastable=False):
        """ Compute set or accumulate `self.grad` with `grad`, and pass `self.creator.backward(grad)`.
            In effect, calling `self.backward()` will trigger a "back-propagation" from `self` through
            the preceding nodes in the computational graph. Thus a node, `a`, will have the attribute
            `self.grad` return the total derivative d(self)/da.

            Parameters
            ----------
            grad : Optional[float, array_like]
                The value of the incoming derivative. If self.grad is None, it is set to `grad`,
                otherwise its value is added with `grad`.

            _broadcastable : bool, optional (default:False)
                Devs-only: Indicates whether or not the up-stream op
                can utilize broadcasting.

            Raises
            ------
            InvalidNonScalarBackprop
                The configuration of the computational graph is such that `self` must be a 0D tensor
                (i.e. scalar) to invoke self.backprop(grad)."""

        if grad is not None:
            grad = np.asarray(grad.data if isinstance(grad, Tensor) else grad)

            if _broadcastable:
                grad = reduce_broadcast(grad, self.shape)
        else:
            if self.ndim > 0 and self.scalar_only:
                raise Exception("Invalid Backprop: scalar-only violation")

            grad = np.ones(self.shape, dtype=float) if self.ndim > 0 else np.asarray(1)

        self.grad = np.asarray(grad if self.grad is None else self.grad + grad)

        if self._creator is not None:
            self._creator.backward(grad, _broadcastable=isinstance(self._creator, BroadcastableOp))

    def null_gradients(self):
        self.grad = None
        self._ops = []
        if self._creator is not None:
            self._creator.null_gradients()

    @property
    def scalar_only(self):
        """ Indicates whether or not `self.ndim` must be 0 in order to invoke `self.backprop()`.

            Returns
            -------
            bool"""
        return self._scalar_only

    @property
    def constant(self):
        """ A constant will not facilitate back-propagation at its node in the computational graph.

            Returns
            -------
            bool """
        return self._constant

    @property
    def creator(self):
        """ The `Operation` instance that produced `self`.

            Returns
            -------
            pygrad.Operation
            """
        return self._creator

    def reshape(self, *shape):
        """ Returns a tensor with a new shape, without changing its data.

            Parameters
            ----------
            newshape : Union[int, Tuple[int, ...]]
                The new shape should be compatible with the original shape. If
                an integer, then the result will be a 1-D array of that length.
                One shape dimension can be -1. In this case, the value is
                inferred from the length of the array and remaining dimensions.

            Returns
            -------
            Tensor

            Notes
            -----
            `reshape` utilizes C-ordering, meaning that it reads & writes elements using
            C-like index ordering; the last axis index changing fastest, and, proceeding
            in reverse order, the first axis index changing slowest. """
        if hasattr(shape[0], "__iter__"):
            if len(shape) > 1:
                raise TypeError("an integer is required")
            shape = shape[0]
        return self._op(Reshape, self, op_args=(shape,))

    def __add__(self, other):
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self._op(Add, other, self)

    def __sub__(self, other):
        return self._op(Subtract, self, other)

    def __rsub__(self, other):
        return self._op(Subtract, other, self)

    def __truediv__(self, other):
        return self._op(Divide, self, other)

    def __rtruediv__(self, other):
        return self._op(Divide, other, self)

    def __mul__(self, other):
        return self._op(Multiply, self, other)

    def __rmul__(self, other):
        return self._op(Multiply, other, self)

    def __pow__(self, other):
        return self._op(Power, self, other)

    def __rpow__(self, other):
        return self._op(Power, other, self)

    def __iadd__(self, other):
        return self + other

    def __imul__(self, other):
        return self * other

    def __isub__(self, other):
        return self - other

    def __idiv__(self, other):
        return self / other

    def __neg__(self):
        return self._op(Negative, self)

    def __repr__(self):
        if self.data.ndim == 0:
            return "Tensor({})".format(self.data.item())
        elif self.data.ndim == 1:
            return "Tensor({})".format(self.data)
        else:
            return "Tensor(\n{}\n)".format(self.data)

    def __lt__(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return self.data < value

    def __le__(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return self.data <= value

    def __gt__(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return self.data > value

    def __ge__(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return self.data >= value

    def __eq__(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return self.data == value

    def __ne__(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return self.data != value

    def __pos__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __copy__(self):
        """ Produces a copy of self with copy.creator=None"""
        return Tensor(np.copy(self.data), _creator=None, constant=self.constant, _scalar_only=self.scalar_only)

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __getitem__(self, item):
        return self._op(GetItem, self, op_args=(item,))

    def __setitem__(self, key, value):
        if self.constant:
            raise ValueError("Tensor constants do not support __setitem__")

        # old_tensor is the tensor pre-setitem
        old_tensor = self.__copy__()
        old_tensor._creator = self.creator
        old_tensor._ops = self._ops

        # point all ops involving `self` to old_tensor instead
        for op in old_tensor._ops:
            for i in range(len(op.variables)):
                if op.variables[i] is self:
                    op.variables = op.variables[:i] + (old_tensor,) + op.variables[i+1:]
                    break

        # self becomes the tensor post-setitem
        out = self._op(SetItem, old_tensor, value, op_args=(key,))
        self._creator = out.creator
        self._scalar_only = out.scalar_only
        self._ops = out._ops
        self.data = out.data


    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def sum(self, axis=None, keepdims=False):
        """ Sum of tensor elements over a given axis.
            Parameters
            ----------
            axis : Optional[int, Tuple[ints, ...]
                Axis or axes along which a sum is performed.  The default,
                axis=None, will sum all of the elements of the input tensor.  If
                axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, a sum is performed on all of the axes
                specified in the tuple instead of a single axis or all the axes as
                before.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input tensor.
            Returns
            -------
            sum_along_axis : Tensor
                A Tensor with the same shape as `self`, with the specified
                axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
                a 0-dim Tensor is returned."""
        return self._op(Sum, self, op_args=(axis, keepdims))

    def mean(self, axis=None, keepdims=False):
        """ Mean of tensor elements over a given axis.

            Parameters
            ----------
            axis : Optional[int, Tuple[ints, ...]
                Axis or axes along which a mean is performed.  The default,
                axis=None, will mean all of the elements of the input tensor.  If
                axis is negative it counts from the last to the first axis.

                If axis is a tuple of ints, a mean is performed on all of the axes
                specified in the tuple instead of a single axis or all the axes as
                before.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input tensor.

            Returns
            -------
            mean_along_axis : Tensor
                A Tensor with the same shape as `self`, with the specified
                axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
                a 0-dim Tensor is returned."""
        return self._op(Mean, self, op_args=(axis, keepdims))

    def max(self, axis=None, keepdims=False):
        """ Return the maximum of a tensor, or along its axes.

            Parameters
            ----------
            axis : Optional[int, Tuple[int, ...]]
                Axis or axes along which to operate. By default, flattened input is used.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `arr`.

            Returns
            -------
            max : Tensor
                Maximum of `a`. If `axis` is None, the result is a 0-D tensor."""
        return self._op(MaxMin, self, op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin='max'))

    def min(self, axis=None, keepdims=False):
        """ Return the minimum of a tensor, or along its axes.

            Parameters
            ----------
            axis : Optional[int, Tuple[int, ...]]
                Axis or axes along which to operate. By default, flattened input is used.

            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the original `arr`.

            Returns
            -------
            min : Tensor
                Minimum of `a`. If `axis` is None, the result is a 0-D tensor."""
        return self._op(MaxMin, self, op_kwargs=dict(axis=axis, keepdims=keepdims, maxmin='min'))

    def transpose(self, axes=None):
        """ Permute the dimensions of an array.

            Parameters
            ----------
            axes : list of ints, optional
                By default, reverse the dimensions, otherwise permute the axes
                according to the values given.

            Returns
            -------
            Tensor
                `self` with its axes permuted.  A new tensor is returned. """
        return self._op(Transpose, self, op_args=(axes,))

    @property
    def T(self):
        """ Same as self.transpose(), except that self is returned if self.ndim < 2 and
            a view of the underlying data is utilized whenever possible.

            Returns
            -------
            Tensor"""
        return self._op(Tensor_Transpose_Property, self)
