from .operations import *
import numpy as np

__all__ = ['Tensor']


class Tensor(object):
    """ A numpy.array-like object capable of serving as a node in a computational graph that
        supports back-propagation of derivatives via the chain rule."""
    __array_priority__ = 15.0

    def __init__(self, x, *, constant=False, dtype=None, _scalar_only=False, _creator=None):
        """ Parameters
            ----------
            x : array_like
                Input data, in any form that can be converted to an array.  This
                includes numbers, lists, lists of tuples, tuples, tuples of tuples, tuples
                of lists and ndarrays.

            constant : bool, optional (default=False)
                If True, this node is treated as a constant, and thus does not facilitate
                back propagation; `self.grad` will always return `None`.

            **kwargs
                Arguments used internally by PyGrad to

                _scalar_only : bool, optional (default=False)
                    Signals that self.backward() can only be invoked if self.ndim == 0.

                _creator: Optional[pygrad.Operation]
                    The operation whose result is the Tensor `self`.

            """
        assert isinstance(constant, bool)
        self._scalar_only = _scalar_only
        self._creator = _creator

        if isinstance(x, Tensor):
            self.data = x.data
        else:
            self.data = np.asarray(x)
            if dtype is None:
                self._check_valid_dtype(self.data.dtype)
        if dtype is not None:
            self.data = self.data.astype(dtype)
            self._check_valid_dtype(dtype)

        self.grad = None
        self._constant = constant

    @staticmethod
    def _check_valid_dtype(dtype):
        if not np.issubdtype(dtype, np.number):
            raise TypeError("Tensor data must be a numeric type")

    @classmethod
    def _op(cls, Op, a, b, *args, **kwargs):
        """ Wraps bivariate operations performed between tensors: f(a, b).

            Parameters
            ----------
            Op : Operation
                Bivariate operation to be performed using `a` and `b`.

            a : Tensor
                First operand.

            b : Tensor
                Second Operand

            *args
                Arbitrary positional arguments passed to the operation.

            **kwargs
                Arbitrary keyword arguments passed to the operation.

            Returns
            -------
            Tensor
                The tensor-result of the operation."""
        op_const = kwargs.pop("constant", False)
        op_dtype = kwargs.pop("dtype", None)

        if not isinstance(a, cls):
            a = cls(a, constant=True)

        if b is not None and not isinstance(b, cls):
            b = cls(b, constant=True)

        f = Op()

        if not a.constant:
            a._ops.append(f)

        if b is not None:
            if not b.constant:
                b._ops.append(f)
            op_out = f(a, b, *args, **kwargs)
        else:
            op_out = f(a, *args, **kwargs)

        if op_dtype is not None:
            op_out = op_out.astype(op_dtype)
            cls._check_valid_dtype(op_dtype)

        is_const = op_const or (a.constant and (b is None or b.constant))
        scalar_only = f.scalar_only and not is_const
        scalar_only = scalar_only or (a.scalar_only and not a.constant)
        scalar_only = scalar_only or (b is not None and (b.scalar_only and not b.constant))
        return cls(op_out, constant=is_const, _creator=f, _scalar_only=scalar_only)

    @classmethod
    def _mon_op(cls, Op, a, *args, **kwargs):
        return cls._op(Op, a, None, *args, **kwargs)

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

    def _backward(self, grad=None):
        if grad is None:
            grad = np.asarray(1) if self.ndim == 0 else np.ones(self.shape)

        self.grad = np.asarray(grad if self.grad is None else self.grad + grad)
        if self._creator is not None:
            self._creator.backward(grad)

    def backward(self, grad=None):
        """ Compute set or accumulate `self.grad` with `grad`, and pass `self.creator.backward(grad)`.

            In effect, calling `self.backward()` will trigger a "back-propagation" from `self` through
            the preceding nodes in the computational graph. Thus a node, `a`, will have the attribute
            `self.grad` return the total derivative d(self)/da.

            Parameters
            ----------
            grad : Optional[float, array_like]
                The value of the incoming derivative. If self.grad is None, it is set to `grad`,
                otherwise its value is added with `grad`.

            Raises
            ------
            InvalidNonScalarBackprop
                The configuration of the computational graph is such that `self` must be a 0D tensor
                (i.e. scalar) to invoke self.backprop(grad)."""


        if grad is not None:
            grad = np.asarray(grad.data if isinstance(grad, Tensor) else grad)

        if self.scalar_only and self.ndim > 0:
            msg = "A forward-pass operation was called which requires that .backward be called from a scalar object"
            raise InvalidNonScalarBackprop(msg)

        self._backward(grad if grad is None else np.asarray(grad))

    def null_gradients(self):
        self.grad = None
        if self._creator is not None:
            self._creator.null_gradients()

    def __add__(self, other):
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self._op(Add, other, self)

    def __mul__(self, other):
        return self._op(Multiply, self, other)

    def __rmul__(self, other):
        return self._op(Multiply, other, self)

    def __neg__(self):
        return -1 * self

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

    def __invert__(self):
        return -1 * self

    def __len__(self):
        return len(self.data)

    def __copy__(self):
        """ Produces a copy of self with copy.creator=None"""
        return Tensor(np.copy(self.data), _creator=None, constant=self.constant, _scalar_only=self.scalar_only)

    def __contains__(self, item):
        return self.data.__contains__(item)

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

    @property
    def T(self):
        """ Same as self.transpose(), except that self is returned if self.ndim < 2 and
            a view of the underlying data is utilized whenever possible.

            Returns
            -------
            Tensor"""
        return self._mon_op(Tensor_Transpose_Property, self)

    def as_constant(self):
        """ Return `self` as a constant tensor. """
        return Tensor(self, constant=True)