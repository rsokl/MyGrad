""" This module defines the base tensor class along with all of its essential
    attributes and special methods. Public math methods, e.g. ``sum``, ``mean``,
    etc., are bound to the Tensor class in ``mygrad.__init__.py``.
    """

from functools import wraps

from mygrad.math.arithmetic.ops import *
from mygrad.tensor_manip.transpose_like.ops import Tensor_Transpose_Property
from mygrad.tensor_core_ops.indexing import GetItem, SetItem
from mygrad.linalg.ops import MatMul
from mygrad.operation_base import BroadcastableOp
from mygrad._utils import reduce_broadcast

import numpy as np


__all__ = ['Tensor']


class Tensor:
    """ A numpy-array-like object capable of serving as a node in a computational
        graph that supports back-propagation of derivatives via the chain rule.
        See the Examples section of the docstring for more details.

        Like the numpy array, mygrad's tensor stores data as an N-dimensional array
        and provides an interface accessing, setting, and performing vectorized
        operations along the various dimensions of this array. Vectorized operations
        support numpy-style broadcasting semantics.

        The contents of a tensor can be accessed and written to using all variety
        of basic and advanced indexing (along with mixtures of the two).

        Creating a Tensor
        -----------------
        ``mygrad.Tensor`` can be passed any "array-like" object of numerical data.
        This includes numbers, sequences (e.g. lists), nested sequences, numpy-ndarrays,
        and other mygrad-tensors. mygrad also provides familiar numpy-style tensor-creation
        functions (e.g. ``mygrad.arange``, ``mygrad.linspace``, etc.)

        >>> import mygrad as mg
        >>> mg.Tensor(2.3)  # creating a 0-dimensional tensor
        Tensor(2.3)
        >>> mg.Tensor(np.array([1.2, 3.0]))  # casting a numpy-array to a tensor
        Tensor([1.2, 3.0])
        >>> mg.Tensor([[1, 2], [3, 4]])  # creating a 2-dimensional tensor
        Tensor([[1, 2],
                [3, 4]])
        >>> mg.arange(4)    # using numpy-style tensor creation functions
        Tensor([0, 1, 2, 3])


        Forward and Back-Propagation
        ----------------------------
        Let's construct a computational graph consisting of two zero-dimensional
        tensors, ``x`` and ``y``, which are used to compute an output tensor,
        ``f``. This is a "forward pass imperative" style for creating a computational
        graph - the graph is constructed as we carry out the forward-pass computation.

        >>> x = Tensor(3.0)
        >>> y = Tensor(2.0)
        >>> f = 2 * x + y ** 2

        Invoking ``f.backward()`` signals the computational graph to
        compute the total-derivative of ``f`` with respect to each one of its dependent
        variables. I.e. ``x.grad`` will store ``df/dx`` and ``y.grad`` will store
        ``df/dy``. Thus we have back-propagated a gradient from ``f`` through our graph.

        Each tensor of derivatives is computed elementwise. That is, if `x = Tensor(x0, x1, x2)`,
        then df/dx represents `[df/d(x0), df/d(x1), df/d(x2)]`

        >>> f.backward()  # computes df/dx and df/dy
        >>> x.grad  # df/dx
        array(6.0)
        >>> y.grad  # df/dy
        array(4.0)
        >>> f.grad
        array(1.0)  # df/df

        Before utilizing ``x`` and ``y`` in a new computational graph, you must
        'clear' their stored derivative values. ``f.null_gradients()`` signals
        ``f`` and all preceding tensors in its computational graph to clear their
        derivatives.

        >>> f.null_gradients()
        >>> x.grad is None and y.grad is None and f.grad is Nonw
        True

        Accessing the Underlying NumPy Array
        ------------------------------------
        ``mygrad.Tensor`` is a thin wrapper on ``numpy.ndarray``. A tensor's
        underlying numpy-array can be accessed via ``.data``:

        >>> x = mg.Tensor([1, 2])
        >>> x.data
        array([1, 2])

        **Do not modify this underlying array**. Any in-place modifications made to this
        array will not be tracked by any computational graph involving that tensor, thus
        back-propagation through that tensor will likely be incorrect."""
    __array_priority__ = 15.0

    def __init__(self, x, *, dtype=None, constant=False, _scalar_only=False, _creator=None):
        """ Parameters
            ----------
            x : array_like
                Input data, in any form that can be converted to an array.  This
                includes numbers, sequences, nested sequences, numpy-ndarrays,
                and mygrad-tensors.

            Keyword-Only Arguments
            ----------------------
            dtype : Optional[type]
                `int`, `float`, or a real-valued numpy data type. By default the
                data type is inferred from ``x`` via ``numpy.asarray(x)``.

            constant : bool, optional (default=False)
                If True, this node is treated as a constant, and thus does not facilitate
                back propagation; `self.grad` will always return `None`.

            _scalar_only : bool, optional (default=False)
                Signals that self.backward() can only be invoked if self.ndim == 0.
                Should not be set manually by users.

            _creator: Optional[mygrad.Operation]
                The operation-instance whose forward pass produced `self`. Should not
                be set manually by users.
            """
        assert isinstance(constant, bool)
        self._scalar_only = _scalar_only
        self._creator = _creator

        if isinstance(x, Tensor):
            self.data = x.data
        else:
            self.data = np.asarray(x, dtype=dtype)
            self._check_valid_dtype(self.data.dtype)

        self.grad = None
        self._constant = constant

        # used for setitem
        self._ops = []  # Operation instances that utilized self an input tensor

    @staticmethod
    def _check_valid_dtype(dtype):
        if not np.issubdtype(dtype, np.number):
            raise TypeError("Tensor data must be a numeric type, received {}".format(dtype))

    @classmethod
    def _op(cls, Op, *input_vars, op_args=None, op_kwargs=None, constant=False):
        """ Wraps operations performed between tensors: f(a, b, ...).

            Parameters
            ----------
            Op : mygrad.operation_base.Operation
                Operation-class, used to perform forward-pass on `input_vars`.

            input_vars : array_like
                An arbitrary number of input-tensors. These can take any form that
                can be converted to an array.  This includes numbers, sequences, nested
                numerical sequences, numpy-ndarrays, and mygrad-tensors.

            op_args : Optional[Tuple[Any]]
                Arbitrary positional arguments passed to the operation's forward pass.

            op_kwargs : Optional[Dict[str, Any]]
                Arbitrary keyword arguments passed to the operation's forward pass.

            constant : bool, optional (default=False)
                If True, the resulting Tensor is a constant.

            Returns
            -------
            mygrad.Tensor
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

        if isinstance(f, BroadcastableOp) and not f.scalar_only:
            # if broadcasting occurred: scalar-only -> True
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
            grad : Optional[array_like]
                The value of the incoming derivative. If self.grad is None, it is set to `grad`,
                otherwise its value is added with `grad`.

            _broadcastable : bool, optional (default:False)
                Devs-only: Indicates whether or not the up-stream operation
                can utilize broadcasting.

            Raises
            ------
            Exception
                The configuration of the computational graph is such that `self` must be a 0D tensor
                (i.e. scalar) to invoke self.backward()."""

        if grad is not None:
            grad = np.asarray(grad.data if isinstance(grad, Tensor) else grad)

            if _broadcastable:
                grad = reduce_broadcast(grad, self.shape)
        else:
            if self.ndim > 0 and self.scalar_only:
                raise Exception("Invalid Backprop: backpropagation must be triggered by a scalar for this computational graph")

            dtype = float if np.issubdtype(self.dtype, np.signedinteger) else self.dtype
            grad = np.ones(self.shape, dtype=dtype) if self.ndim > 0 else np.asarray(1., dtype=dtype)

        assert grad.shape == self.shape, "A tensor and its associated gradient must possess the same shape"
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
        """ Indicates whether or not `self.ndim` must be 0 in order to invoke `self.backward()`.

            Returns
            -------
            bool"""
        return self._scalar_only

    @property
    def constant(self):
        """ If `True`, this tensor is a constant - it will not propagate any gradient.

            Returns
            -------
            bool """
        return self._constant

    @property
    def creator(self):
        """ The `Operation` instance that produced `self`.

            Returns
            -------
            mygrad.Operation
            """
        return self._creator

    def __len__(self):
        return len(self.data)
    
    def __contains__(self, item):
        return self.data.__contains__(item)

    def __getitem__(self, item):
        return self._op(GetItem, self, op_args=(item,))

    def __setitem__(self, key, value):
        if self.constant and (not isinstance(value, Tensor) or value.constant):
            self.data[key] = value.data if isinstance(value, Tensor) else value
            return None

        # old_tensor is the tensor pre-setitem
        old_tensor = Tensor(self, constant=self.constant, _scalar_only=self.scalar_only, _creator=self.creator)
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
        self._constant = out.constant

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
    
    def __matmul__(self, other):
        return self._op(MatMul, self, other)

    def __rmatmul__(self, other):
        return self._op(MatMul, other, self)

    def __pow__(self, other):
        return self._op(Power, self, other)

    def __rpow__(self, other):
        return self._op(Power, other, self)

    def __neg__(self):
        return self._op(Negative, self)

    def __pos__(self):
        return self

    def __repr__(self):
        return repr(self.data).replace("array", "Tensor").replace("\n", "\n ")

    def __copy__(self):
        """ Produces a copy of self with copy.creator=None"""
        return Tensor(np.copy(self.data), _creator=None, constant=self.constant, _scalar_only=self.scalar_only)

    def item(self):
        """ Copy an element of a tensor to a standard Python scalar and return it.

            Returns
            -------
            z : Standard Python scalar object
                A copy of the specified element of the tensor as a suitable
                Python scalar"""
        if self.size > 1:
            raise ValueError("can only convert a tensor of size 1 to a Python scalar")
        return self.data.item()

    def __float__(self):
        if self.size > 1:
            raise TypeError("can only convert a tensor of size 1 to a Python scalar")
        return float(self.data)

    def __int__(self):
        if self.size > 1:
            raise TypeError("can only convert a tensor of size 1 to a Python scalar")
        return int(self.data)

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
        return self._op(Tensor_Transpose_Property, self)


# set all comparison operators - mirrors ndarray methods
def tensor_to_array_wrapper(func):
    @wraps(func)
    def wrapped(x, y):
        return func(x.data, y.data if isinstance(y, Tensor) else y)
    return wrapped


for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
    setattr(Tensor, op, tensor_to_array_wrapper(getattr(np.ndarray, op)))
