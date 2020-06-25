"""
This module defines the base tensor class along with all of its essential
attributes and special methods. Public math methods, e.g. ``sum``, ``mean``,
etc., are bound to the Tensor class in ``mygrad.__init__.py``.
"""

from functools import wraps
from typing import Optional, Set, Type, Union

import numpy as np

from mygrad._utils import is_invalid_gradient
from mygrad.errors import InvalidBackprop, InvalidGradient
from mygrad.linalg.ops import MatMul
from mygrad.math.arithmetic.ops import (
    Add,
    Divide,
    Multiply,
    Negative,
    Positive,
    Power,
    Subtract,
)
from mygrad.operation_base import BroadcastableOp, Operation
from mygrad.tensor_core_ops.indexing import GetItem, SetItem
from mygrad.tensor_manip.array_shape.ops import Flatten, Reshape
from mygrad.tensor_manip.transpose_like.ops import Tensor_Transpose_Property

__all__ = ["Tensor"]


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
    >>> x.grad is None and y.grad is None and f.grad is None
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

    def __init__(
        self, x, *, dtype=None, constant=False, _scalar_only=False, _creator=None
    ):
        """
        Parameters
        ----------
        x : array_like
            Input data, in any form that can be converted to an array.  This
            includes numbers, sequences, nested sequences, numpy-ndarrays,
            and mygrad-tensors.

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
        self._creator = _creator  # type: Union[None, Operation]

        if isinstance(x, Tensor):
            self.data = x.data
        else:
            self.data = np.asarray(x, dtype=dtype)
            self._check_valid_dtype(self.data.dtype)

        self.grad = None  # type: Union[None, np.ndarray]
        self._constant = constant

        # track all operations that this tensor participates in
        self._ops = set()  # type: Set[Operation]

        # track the operations that have contributed to this tensor's gradient during a back-prop
        self._accum_ops = set()  # type: Set[Operation]

    def astype(
        self, dtype: Union[type, str], *, constant: Optional[bool] = None
    ) -> "Tensor":
        """Returns a distinct tensor with its data modified to have the specified
        data type.

        The resulting tensor does not belong to any pre-existing computation graph; i.e.
        it is as if this tensor was created 'from scratch'.

        Parameters
        ----------
        dtype : Union[type, str]
            The real-valued numeric data type. This can be a numpy dtype or
            a corresponding string identifier.

        constant : Optional[bool]
            If specified, determines if the returned tensor is a constant.
            Otherwise this argument is inferred from the original tensor.

        Returns
        -------
        Tensor
            The resulting tensor with the specified data type.

        Examples
        --------
        >>> import mygrad as mg
        >>> import numpy as np
        >>> x = mg.arange(10); x
        Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        Using a string to specify the data type:

        >>> x.astype("float32")
        Tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)

        Specifying a numpy data type object, and specifying that the
        tensor is to be treated as a constant:

        >>> x.astype(np.int8, constant=True)
        Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int8)
        """
        constant = constant if constant is not None else self.constant
        return type(self)(self.data.astype(dtype), constant=constant)

    @staticmethod
    def _check_valid_dtype(dtype):
        if not np.issubdtype(dtype, np.number):
            raise TypeError(f"Tensor data must be a numeric type, received {dtype}")

    @classmethod
    def _op(
        cls,
        Op: Type[Operation],
        *input_vars,
        op_args=None,
        op_kwargs=None,
        constant=False,
    ):
        """Wraps operations performed between tensors: f(a, b, ...).

        For developer use only.

        Parameters
        ----------
        Op : Type[Operation]
            Operation-class, used to perform forward-pass on `input_vars`.

        input_vars : Tuple[array_like, ...]
            An arbitrary number of input-tensors. These can take any form that
            can be converted to an array.  This includes numbers, sequences, nested
            numerical sequences, numpy-ndarrays, and mygrad-tensors.

        op_args : Optional[Tuple[Any, ...]]
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

        tensor_vars = tuple(
            cls(var, constant=True) if not isinstance(var, cls) else var
            for var in input_vars
        )

        is_const = constant or all(var.constant for var in tensor_vars)

        f = Op()
        f.graph = {f}
        f.graph.update(
            *(
                var._creator.graph
                for var in tensor_vars
                if var._creator is not None and not var.constant
            )
        )
        op_out = f(*tensor_vars, *op_args, **op_kwargs)

        if isinstance(f, BroadcastableOp) and not f.scalar_only:
            # if broadcasting occurred: scalar-only -> True
            f.scalar_only = any(
                op_out.shape != i.shape for i in tensor_vars if not i.constant
            )

        if not is_const:
            # record that a variable participated in that op
            for var in tensor_vars:
                if not var.constant:
                    var._ops.add(f)

        scalar_only = f.scalar_only and not is_const
        for var in tensor_vars:
            scalar_only = scalar_only or (var.scalar_only and not var.constant)

        return cls(op_out, constant=is_const, _creator=f, _scalar_only=scalar_only)

    def backward(self, grad=None):
        """ Compute set or accumulate ``self.grad`` with `grad`, and pass ``self.creator.backward(grad)``.
        In effect, calling ``self.backward()`` will trigger a "back-propagation" from ``self`` through
        the preceding nodes in the computational graph. Thus a node, ``a``, will have the attribute
        ``self.grad`` return the total derivative `d(self)/da`.

        Parameters
        ----------
        grad : Optional[array_like]
            The value of the incoming derivative. If self.grad is None, it is set to `grad`,
            otherwise its value is added with `grad`.

        Raises
        ------
        Exception
            The configuration of the computational graph is such that ``self`` must be a 0D tensor
            (i.e. scalar) to invoke ``self.backward()``.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor(2)
        >>> y = mg.Tensor(3)
        >>> w = x * y
        >>> f = 2 * w
        >>> f.backward()  # computes df/df, df/dw, df/dy, and df/dx

        >>> f.grad  # df/df == 1 by identity
        array(1.)
        >>> w.grad  # df/dw
        array(2.)
        >>> y.grad # df/dy = df/dw * dw/dy
        array(4.)
        >>> x.grad # df/dx = df/dw * dw/dx
        array(6.)
        """
        if self._constant:
            return

        if grad is not None:
            self.grad = np.asarray(grad.data if isinstance(grad, Tensor) else grad)
            if is_invalid_gradient(self.grad):
                raise InvalidGradient(
                    f"An invalid gradient-value was passed to "
                    f"\n\t`{type(self).__name__}.backward(<gradient>)`"
                    f"\nGradients are expected to be real-valued scalars or "
                    f"numpy arrays, got a gradient of type: {type(grad)}"
                )

        else:
            if self.ndim > 0 and self._scalar_only:
                raise InvalidBackprop(
                    "Backpropagation must be invoked from a "
                    "scalar-tensor (a 0D tensor) for this computational "
                    "graph."
                )
            dtype = float if np.issubdtype(self.dtype, np.signedinteger) else self.dtype
            self.grad = (
                np.ones(self.shape, dtype=dtype)
                if self.ndim > 0
                else np.asarray(1.0, dtype=dtype)
            )

        if self.creator is not None:
            self._backward(graph=self.creator.graph)

    def _backward(self, *, graph):
        """
        **For dev-use only**

        If `self` has accumulated incoming gradients from all operations in the terminal node's
        computational graph, back-propagate the accumulated gradient to the creator of `self`.

        Parameters
        ----------
        graph : Set[Operation]
            The set of all operations relevant to the terminal node of the computational graph,
            which triggered back-propagation

        Raises
        ------
        AssertionError
            Raises if the tensor and its associated gradient possess different shapes.
            Raises if `_backward` triggered on a tensor with gradient of `None`.
        """
        assert self.grad is not None, (
            "backprop, post grad-accumulation, was triggered "
            "on a tensor with no gradient"
        )
        assert self.grad.shape == self.shape, (
            f"A tensor and its associated gradient must possess the same shape. Got:"
            f"\ntensor-shape: {self.shape}"
            f"\ngrad-shape: {self.grad.shape}"
        )
        if self._creator is not None and not bool(
            graph & (self._ops - self._accum_ops)
        ):
            self._accum_ops.clear()
            self._creator.backward(self.grad, graph=graph)

    def null_gradients(self, clear_graph=True):
        """
        Sets the gradient for this tensor and for all preceding tensors in the computation graph
        to ``None``.

        Additionally, the computational graph that terminates in this tensor can also be cleared
        during this process.

        Parameters
        ----------
        clear_graph : bool, optional (default=True)
            If ``True`` clear the computational graph in addition to nulling the gradients.

        Notes
        -----
        It is advised to clear the computational graph when nulling gradients, i.e. invoke
        ``null_gradients(clear_graph=True)`` (or simply ``null_gradients()``). This de-references
        all intermediate operations and tensors in the computational graph and thus permits
        garbage collection - freeing the memory that was used by the computational graph.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor(2)
        >>> y = mg.Tensor(3)
        >>> w = x * y
        >>> f = 2 * w
        >>> f.backward()  # computes df/df, df/dw, df/dy, and df/dx
        >>> any(tensor.grad is None for tensor in (f, w , x, y))
        False

        >>> f.null_gradients()  # set tensor.grad to None for all tensors in the graph
        >>> all(tensor.grad is None for tensor in (f, w , x, y))
        True
        """
        self._null_gradients(clear_graph=clear_graph, seen=set())

    def _null_gradients(self, *, clear_graph: bool, seen: Set["Tensor"]):
        """
        Nulls gradients using depth-first graph traversal

        Parameters
        ----------
        clear_graph : bool, optional (default=True)
            If ``True`` clear the computational graph in addition to nulling the gradients.

        seen : Set[Tensor]
            The set of all Tensors already visited during null-gradients traversal"""
        self.grad = None

        if self._creator is None:
            return

        if not clear_graph:
            assert isinstance(seen, set)

            if self in seen:
                return

        creator = self._creator

        if clear_graph:
            # marks tensor as "visited" during clear-graph traversal
            self._creator = None
        else:
            # marks tensor as "visited" during null-gradients graph traversal
            seen.add(self)

        for var in creator.variables:  # type: Tensor
            if clear_graph:
                var._ops.clear()
            var._null_gradients(clear_graph=clear_graph, seen=seen)

    def clear_graph(self):
        """
        Clear the computational graph for all of the nodes preceding this tensor.
        """
        if self._creator is None:
            return

        creator = self._creator
        self._creator = None  # marks tensor as "visited" during graph-traversal

        for var in creator.variables:  # type: Tensor
            var._ops.clear()
            var.clear_graph()

    @property
    def scalar_only(self):
        """ Indicates whether or not `self.ndim` must be 0 in order to invoke `self.backward()`.

        E.g. a computational graph that involves a broadcast-multiplication of a non-constant
        tensor can only support back-propagation from a scalar. Otherwise the gradient associated
        with each element of a given tensor would, itself, have to be represented by a high-dimensional
        tensor. MyGrad only supports computational graphs in which a tensor's gradient has the same
        shape as the tensor itself.

        Returns
        -------
        bool

        Notes
        -----
        In the following example, see that, because ``x`` was broadcasted to a
        shape-(2, 3) tensor, the derivative :math:`df/dx` could not be a shape-(3,) tensor.
        Each element of ``x`` affects two entries of ``z``, thus :math:`df/dx`
        would have to be a shape-(2, 3) array. Therefore ``z.scalar_only`` returns ``True``.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor([1., 2., 3.])  # shape-(3,)
        >>> y = mg.Tensor([[4., 1., 9.], # shape-(2, 3)
        ...                [0., 2., 8.]])
        >>> z = x * y  # uses numpy-style broadcasting
        >>> z.scalar_only
        True
        """
        return self._scalar_only

    @property
    def constant(self):
        """ If ``True``, this tensor is a constant; it will not propagate any gradient.

        Additionally, any tensor that is a descendant of constant tensors will also
        be a constant.

        Python scalars and NumPy arrays are treated as constant tensors when included
        in MyGrad computational graphs.

        Returns
        -------
        bool

        Examples
        --------
        Constant-tensors do not back-propagate gradients:

        >>> import mygrad as mg
        >>> x = mg.Tensor([1., 2.], constant=True)
        >>> y = mg.Tensor([0., 3.], constant=False)
        >>> f = x * y
        >>> f.backward()

        >>> x.grad is None  # x has no gradient
        True
        >>> y.grad
        array([1., 2.])

        A tensor that is derived solely from constant tensors is also
        a constant:

        >>> import numpy as np
        >>> x = mg.Tensor([1., 2.], constant=True)
        >>> y = mg.Tensor([0., 3.], constant=True)
        >>> z = (x + y) ** 2 - np.array([8., 7.])
        >>> z.constant
        True
        """
        return self._constant

    @property
    def creator(self) -> Union[Operation, BroadcastableOp]:
        """ The ``Operation`` instance that produced ``self``.

        Returns
        -------
        Operation

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor(3)
        >>> y = mg.Tensor(2)
        >>> z = x * y  # Multiply(x, y) -> z
        >>> z.creator
         <mygrad.math.arithmetic.ops.Multiply at 0x2df5a130438>
        """
        return self._creator

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __getitem__(self, item):
        # In the same way that numpy doesn't let you iterate over 0-dimensional
        # arrays, don't allow iteration over 0-dimensional arrays.
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return self._op(GetItem, self, op_args=(item,))

    def _mirror_tensor(self, tensor: "Tensor"):
        """ *Dev use only*

        Points all of the attributes of ``self`` to those of
        ``tensor`` so that they reference all of the same data structures.
        This is used to facilitate "in-place" operations.
        """
        self.__dict__ = tensor.__dict__

    def _in_place_op(
        self,
        inplace_op: Type[Operation],
        *input_vars,
        op_args=None,
        op_kwargs=None,
        constant=False,
    ):
        """ A substitute for ``self._op``, to facilitate in-place operations.

        Note that in-place operations are generally less efficient than their
        counterparts due to the additional bookkeeping that is required to
        update the computational graph. The benefit lies purely in convenience
        for the user.
        """
        # old_tensor is the tensor pre-setitem
        old_tensor = Tensor(
            self,
            constant=self.constant,
            _scalar_only=self._scalar_only,
            _creator=self.creator,
        )
        old_tensor._ops = self._ops
        old_tensor._accum_ops = self._accum_ops

        # point all ops involving `self` to old_tensor instead
        for op in old_tensor._ops:
            for i in range(len(op.variables)):
                if op.variables[i] is self:
                    op.variables = (
                        op.variables[:i] + (old_tensor,) + op.variables[i + 1 :]
                    )

        # self becomes the tensor post-setitem
        out = self._op(
            inplace_op,
            old_tensor,
            *input_vars,
            op_args=op_args,
            op_kwargs=op_kwargs,
            constant=constant,
        )
        self._mirror_tensor(out)

    def __setitem__(self, key, value):
        if self.constant and (not isinstance(value, Tensor) or value.constant):
            self.data[key] = value.data if isinstance(value, Tensor) else value
            return None

        # self becomes the tensor post-setitem
        self._in_place_op(SetItem, value, op_args=(key,))

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
        return self._op(Positive, self)

    def __repr__(self):
        return repr(self.data).replace("array", "Tensor").replace("\n", "\n ")

    def __copy__(self):
        """ Produces a copy of ``self`` with ``copy.creator=None``.

        Copies of the underlying numpy data array and gradient array are created.

        Returns
        -------
        Tensor
        """
        copy = Tensor(
            np.copy(self.data),
            _creator=None,
            constant=self.constant,
            _scalar_only=self._scalar_only,
        )
        copy.grad = np.copy(self.grad) if self.grad is not None else None
        return copy

    def copy(self):
        """ Produces a copy of ``self`` with ``copy.creator=None``.

        Copies of the underlying numpy data array and gradient array are created.

        Returns
        -------
        Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor(data, constant=constant)
        >>> y = x * 2
        >>> y.backward()
        >>> y_copy = y.copy()
        >>> y_copy
        Tensor(6)
        >>> y_copy.grad
        array(1.)
        >>> y_copy.creator is None
        True
        """
        return self.__copy__()

    def item(self):
        """ Copy an element of a tensor to a standard Python scalar and return it.

        Note that the returned object does not support back-propagation.

        Returns
        -------
        z : Standard Python scalar object
            A copy of the specified element of the tensor as a suitable
            Python scalar

        Examples
        --------
        >>> import mygrad as mg
        >>> x = Tensor([22.2])
        >>> x.item()
        22.2
        >>> type(x.item())
        float """
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

    def flatten(self, constant=False):
        """ Return a copy of the tensor collapsed into one dimension.

        This docstring was adapted from ``numpy.ndarray.flatten``.

        Returns
        -------
        mygrad.Tensor
            A copy of the input tensor, flattened to one dimension.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Notes
        -----
        To return a flattened view of the tensor, use ``x.reshape(-1)``.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor([[1, 2],
        ...                [3, 4]])
        >>> x.flatten()
        Tensor([1, 2, 3, 4])
        """
        return Tensor._op(Flatten, self, constant=constant)

    @property
    def size(self):
        """
        Number of elements in the tensor. i.e., the product of the tensor's
        dimensions.

        Returns
        -------
        int

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.zeros((3, 5, 2))  # creates a tensor with 3x5x2 (= 30) elements
        >>> x.size
        30
        """
        return self.data.size

    @property
    def ndim(self):
        """ Number of tensor dimensions. I.e. the number
        of indices that must be supplied to uniquely specify
        an element in the tensor.

        Returns
        -------
        int

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor([1, 2, 3])
        >>> x.ndim
        1
        >>> x[0]  # a single index identifies an element in `x`
        Tensor(0)

        >>> y = mg.Tensor([[1, 2, 3],
        ...                [4, 5, 6]])
        >>> y.ndim
        2
        >>> y[0, 0]  # two indices are required to identify an element in `x`
        Tensor(0)"""
        return self.data.ndim

    @property
    def dtype(self):
        """ Data-type of the tensor's elements.

        Returns
        -------
        numpy dtype object

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor([[0, 1],
        ...                [2, 3]])
        >>> x.dtype
        dtype('int32')
        >>> type(x.dtype)
        <type 'numpy.dtype'>"""
        return self.data.dtype

    @property
    def shape(self):
        """ Tuple of tensor dimension-sizes.

        Sizes are reported in row-major order.

        Returns
        -------
        Tuple[int, ...]

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor([1, 2, 3, 4])  # axis-0 has size 4
        >>> x.shape
        (4,)
        >>> y = mg.Tensor([[1, 2, 3],    # axis-0 has size 2, axis-1 has size 3
        ...                [4, 5, 6]])
        >>> y.shape
        (2, 3)

        See Also
        --------
        mygrad.reshape : similar function
        Tensor.reshape : similar method"""
        return self.data.shape

    @shape.setter
    def shape(self, newshape):
        if self.constant:
            self.data.shape = newshape
            return
        self._in_place_op(Reshape, op_args=(newshape,), constant=self.constant)

    def reshape(self, *newshape, constant=False):
        """ Returns a tensor with a new shape, without changing its data.
        This docstring was adapted from ``numpy.reshape``

        Parameters
        ----------
        *newshape : Union[int, Tuple[int, ...]]
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D tensor of that length.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the tensor and remaining dimensions.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor
            ``a`` with its shape changed.  A new tensor is returned.

        Notes
        -----
        ``reshape`` utilizes C-ordering, meaning that it reads & writes elements using
        C-like index ordering; the last axis index changing fastest, and, proceeding
        in reverse order, the first axis index changing slowest.

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> a.reshape(6)
        Tensor([1, 2, 3, 4, 5, 6])

        >>> a.reshape(3, -1))   # the unspecified value is inferred to be 2
        Tensor([[1, 2],
                [3, 4],
                [5, 6]])
            """

        if not newshape:
            raise TypeError("reshape() takes at least 1 argument (0 given)")
        if hasattr(newshape[0], "__iter__"):
            if len(newshape) > 1:
                raise TypeError("an integer is required")
            newshape = newshape[0]
        return Tensor._op(Reshape, self, op_args=(newshape,), constant=constant)

    @property
    def T(self):
        """ Same as self.transpose(), except that self is returned if self.ndim < 2 and
        a view of the underlying data is utilized whenever possible.

        Returns
        -------
        Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> y = mg.Tensor([[1, 2, 3],
        ...                [4, 5, 6]])
        >>> y.T
        Tensor([[1, 4],
                [2, 5],
                [3, 6]])
        """
        return self._op(Tensor_Transpose_Property, self)


# set all comparison operators - mirrors ndarray methods
def tensor_to_array_wrapper(func):
    @wraps(func)
    def wrapped(x, y):
        return func(x.data, y.data if isinstance(y, Tensor) else y)

    return wrapped


for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
    setattr(Tensor, op, tensor_to_array_wrapper(getattr(np.ndarray, op)))
