"""
This module defines the base tensor class along with all of its essential
attributes and special methods. Public math methods, e.g. ``sum``, ``mean``,
etc., are bound to the Tensor class in ``mygrad.__init__.py``.
"""

from functools import wraps
from numbers import Number
from typing import List, Optional, Set, Type, Union

import numpy as np

import mygrad._graph_tracking as _track
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
    Square,
    Subtract,
)
from mygrad.operation_base import BroadcastableOp, Operation
from mygrad.tensor_core_ops.indexing import GetItem, SetItem
from mygrad.tensor_manip.array_shape.ops import Flatten, Reshape
from mygrad.tensor_manip.transpose_like.ops import Tensor_Transpose_Property

__all__ = ["Tensor", "asarray"]


def _is_view_of(parent: "Tensor", child: np.ndarray) -> bool:
    if np.shares_memory(parent, child):
        return True
    elif child.size == 0 and child.base is not None:
        if (child.base is parent.data) or (child.base is parent.data.base):
            return True
    return False


def asarray(a, dtype=None, order=None) -> np.ndarray:
    """Convert the input to an array.

    This docstring is adapted from that of ``numpy.asarray``

    Parameters
    ----------
    a : array_like
        Input data, in any form - including a mygrad tensor - that can be converted to an array. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.

    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.

    Examples
    --------
    Convert a list into an array:

    >>> import mygrad as mg
    >>> a = [1, 2]
    >>> mg.asarray(a)
    array([1, 2])

    Convert a tensor into an array. No copy of the
    underlying numpy array is created:

    >>> t = mg.Tensor([1, 2.])
    >>> mg.asarray(t)
    array([1., 2.])
    >>> t.data is np.asarray(t))
    True

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> mg.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> mg.asarray(a, dtype=np.float32) is a
    True
    >>> mg.asarray(a, dtype=np.float64) is a
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1.0, 2), (3.0, 4)], dtype='f4,i4').view(np.recarray)
    >>> mg.asarray(a) is a
    False
    >>> np.asanyarray(a) is a
    True
    """
    return np.asarray(a, dtype=dtype, order=order)


def astensor(t, dtype=None, constant=None) -> "Tensor":
    """Convert the input to a tensor.

    A tensor `t` is returned unchanged - its gradient and computational
    graph state preserved - if dtype and constant are compatible.
    A copy of the underlying numpy array is created only if dtype is
    incompatible or if a non-constant tensor is being created from a constant.

    Parameters
    ----------
    t : array_like
        Input data, in any form that can be converted to a tensor. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.

    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    constant : Optional[bool]
        By default, `constant` is inferred from `t` if `t` is a tensor,
        otherwise it defaults to `False`.

    Returns
    -------
    out : Tensor
        Tensor interpretation of `a`.  No copy is performed if the input
        is already a tensor with matching dtype and constant-flag.

    Examples
    --------
    Convert a list into an array:

    >>> import mygrad as mg
    >>> import numpy as np
    >>> t = [1, 2]
    >>> mg.astensor(t)
    Tensor([1, 2])

    Convert an array into a tensor. No copy of the
    underlying numpy array is created:

    >>> a = np.array([1, 2.])
    >>> mg.astensor(a)
    Tensor([1., 2.])
    >>> a is mg.astensor(a).data
    True

    Existing tensors are not copied and their gradients and
    computational graphs are preserved:

    >>> t = 2 * mg.Tensor([1, 2])
    >>> t2 = mg.astensor(t)
    >>> t is t2
    True
    >>> t.creator is t2.creator
    True

    If `dtype` is set, a new tensor is created - with copied data - only
    if dtype does not match:

    >>> t = mg.Tensor([1, 2], dtype=np.float32)
    >>> mg.astensor(t, dtype=np.float32) is t
    True
    >>> mg.astensor(t, dtype=np.float64) is t
    False

    Creating a non-constant tensor from a constant tensor will copy
    the underlying data.

    >>> t = mg.Tensor([1, 2], constant=True)
    >>> mg.astensor(t, constant=False).data is t.data
    False

    Otherwise, if `constant` is set, a new tensor is created (with
    no copy of the underlying data) only if constant doesn't match.

    >>> t = mg.Tensor([1, 2], constant=False)
    >>> mg.astensor(t, constant=False) is t
    True
    >>> mg.astensor(t, constant=True) is t
    False
    >>> mg.astensor(t, constant=True).data is t.data
    True
    """
    if (
        isinstance(t, Tensor)
        and (constant is None or t.constant is constant)
        and (dtype is None or (t.dtype == np.dtype(dtype)))
    ):
        return t
    else:
        if constant is None:
            constant = t.constant if isinstance(t, Tensor) else False
        return Tensor(t, dtype=dtype, constant=constant)


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

    Creating a non-constant tensor will copy array data:

    >>> import numpy as np
    >>> arr = np.arange(10.)
    >>> t_var = Tensor(arr, constant=False)
    >>> np.shares_memory(arr, t_var)
    False

    Creating constant tensor will not make a copy of the array data:

    >>> t_const = Tensor(arr, constant=True)
    >>> np.shares_memory(arr, t_const)
    True

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

    Once the gradients are computed, the computational graph containing ``x``,
    ``y``, and ``f`` is cleared automatically. Additionally, involving any
    of these tensors in a new computational graph will automatically null
    their gradients.

    >>> 2 * x
    >>> x.grad is None
    True

    Or, you can use the ``tensor.null_grad()`` method to manually clear a
    tensor's gradient

    >>> y.null_grad()
    Tensor(2.)
    >>> y.grad is None
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
        self,
        x,
        *,
        dtype=None,
        constant=False,
        _scalar_only=False,
        _creator=None,
        _base: Optional["Tensor"] = None,
        _copy_data: Optional[bool] = None,
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

        _creator : Optional[mygrad.Operation]
            The operation-instance whose forward pass produced `self`. Should not
            be set manually by users.

        _base : Optional[Tensor]
            Sets the base tensor that ``self`` is a view of

        _copy_data : Optional[bool]
            Determines if the incoming array-data will be copied
        """
        if not isinstance(constant, bool):
            raise TypeError(f"`constant` must be a boolean value, got: {constant}")

        assert isinstance(_scalar_only, bool)
        assert isinstance(_creator, (Operation, type(None)))
        assert isinstance(_base, (Tensor, type(None)))
        assert isinstance(_copy_data, (bool, type(None)))

        self._scalar_only = _scalar_only
        self._creator = _creator  # type: Union[None, Operation]

        if _copy_data is None:
            _copy_data = not constant

        to_array = np.array if _copy_data else np.asarray
        self.data = to_array(x, dtype=dtype)  # type: np.ndarray
        self._check_valid_dtype(self.data.dtype)

        self.grad = None  # type: Union[None, np.ndarray]
        self._constant = constant

        # track all operations that this tensor participates in
        self._ops = set()  # type: Set[Operation]

        # track the operations that have contributed to this tensor's gradient during a back-prop
        self._accum_ops = set()  # type: Set[Operation]

        # base points to the initial tensor that owns the memory of this
        # tensor
        self._base = _base  # type: Optional[Tensor]
        # stores all of the tensors that are a view of this tensor
        self._view_children = []  # type: List[Tensor]

        # used to track original 'writeable' statuses of array data
        self._data_was_writeable = None  # type: Optional[bool]
        self._base_data_was_writeable = None  # type: Optional[bool]

    def _lock_writeability(self) -> "Tensor":
        """Sets the underlying numpy-array (and its base) to be read-only.

        The initial writeability flags of all involved arrays are recorded so
        that they can be restored via ``self._restore_writeability()``"""
        if self._data_was_writeable is not None or self.base is not None:
            return self

        self._data_was_writeable = self.data.flags.writeable
        self.data.flags.writeable = False

        if self.data.base is not None:
            self._base_data_was_writeable = self.data.base.flags.writeable
            self.data.base.flags.writeable = False

        return self

    def _restore_writeability(self):
        """Restores the writeability flag for the underlying numpy array (and its base)"""
        if self.base is not None:
            # A view of an array inherits the writeability of its parent;
            # however, the writeability of the view is independent of that
            # of its parent. I.e. toggling the parent's writeability does
            # not affect the view's writeability.
            #
            # Furthermore, the parent's writeability must be restored before
            # restoring the view's writeability
            self.base._restore_writeability()
            if self.base.data.flags.writeable:
                self.data.flags.writeable = self.base.data.flags.writeable
            return

        if self._data_was_writeable is None:
            return

        # the base-data's writeability must be restored before that of its view's
        if self._base_data_was_writeable is not None:
            if self._base_data_was_writeable:
                self.data.base.flags.writeable = self._base_data_was_writeable
            self._base_data_was_writeable = None

        if self._data_was_writeable:
            self.data.flags.writeable = self._data_was_writeable
        self._data_was_writeable = None

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
        *input_vars: "Tensor",
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

        # cast all input-vars to tensors
        if _track.TRACK_GRAPH:
            # lock memory of array data and clear any tensor
            # gradients
            tensor_vars = tuple(
                (
                    cls(var, constant=True)
                    if not isinstance(var, Tensor)
                    else var.null_grad()
                )._lock_writeability()
                for var in input_vars
            )
        else:
            # operations are not being tracked - don't lock memory or null grads
            tensor_vars = tuple(
                cls(var, constant=True) if not isinstance(var, Tensor) else var
                for var in input_vars
            )

        if op_args is None:
            op_args = tuple()

        if op_kwargs is None:
            op_kwargs = dict()

        f = Op()

        op_out = f(*tensor_vars, *op_args, **op_kwargs)  # type: np.ndarray

        # Determine whether or not op was a view; if so, `base`
        # points to parent Tensor
        base = None  # type: Optional[Tensor]
        # If output of op is a view - tracks the tensor var that is
        # the parent of the view
        parent_var = None  # type: Optional[Tensor]

        if not f.cannot_return_view:
            vars_can_share_mem = (
                isinstance(var, (np.ndarray, Tensor)) for var in input_vars
            )
            for can_share_mem, var in zip(vars_can_share_mem, tensor_vars):
                if can_share_mem and _is_view_of(parent=var, child=op_out):
                    base = var if var.base is None else var.base
                    parent_var = var
                    break

        if not _track.TRACK_GRAPH:
            # execute operation without tracking creator or any graph
            # information
            return cls(
                op_out,
                constant=constant,  # constant not determined by graph info
                _creator=None,
                _scalar_only=False,
                _base=base,
                _copy_data=False,
            )

        if base is not None:
            # we need to be able to replay view-ops for doing in-place operations
            # on graphs with views
            f.replay_args = op_args
            f.replay_kwargs = op_kwargs
            f.replay_force_constant = constant

        # record graph information
        is_const = constant or all(var.constant for var in tensor_vars)

        f.graph = {f}
        f.graph.update(
            *(
                var._creator.graph
                for var in tensor_vars
                if var._creator is not None and not var.constant
            )
        )

        if isinstance(f, BroadcastableOp) and not f.scalar_only:
            # if broadcasting occurred: scalar-only -> True
            f.scalar_only = any(
                op_out.shape != i.shape for i in tensor_vars if not i.constant
            )

        # record that a variable participated in that op
        for var in tensor_vars:
            var._ops.add(f)

        # determine if node only supports backprop from a scalar
        # terminus
        scalar_only = (f.scalar_only and not is_const) or any(
            var.scalar_only for var in tensor_vars if not var.constant
        )

        out = cls(
            op_out,
            constant=is_const,
            _creator=f,
            _scalar_only=scalar_only,
            _base=base,
            _copy_data=False,
        )._lock_writeability()

        if parent_var is not None:
            parent_var._view_children.append(out)

        return out

    def _replay_op(self, *input_vars) -> "Tensor":
        """ *dev use only*

        Replays the op that produced `self` - called on the specified
        input vars"""
        if self.creator is None:
            raise ValueError(
                "``Tensor._replay_op(...)`` was called on a tensor without a creator."
                "\nPlease report this error at: https://github.com/rsokl/MyGrad/issues"
            )
        return self._op(
            type(self.creator),
            *input_vars,
            op_args=self.creator.replay_args,
            op_kwargs=self.creator.replay_kwargs,
            constant=self.creator.replay_force_constant,
        )

    def backward(self, grad=None):
        """ Compute set or accumulate ``self.grad`` with `grad`, and pass ``self.creator.backward(grad)``.
        In effect, calling ``self.backward()`` will trigger a "back-propagation" from ``self`` through
        the preceding nodes in the computational graph. Thus a node, ``a``, will have the attribute
        ``self.grad`` return the total derivative `d(self)/da`.

        Once back-propagation is finished, the present tensor is removed from all computational
        graphs, and the preceding graph is cleared.

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
        if not _track.TRACK_GRAPH:
            return

        if self.constant:
            self.clear_graph()
            return

        if grad is not None:
            self.grad = asarray(grad)
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

        self.clear_graph()

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
        if self.creator is not None and not bool(graph & (self._ops - self._accum_ops)):
            self._accum_ops.clear()
            self._creator.backward(self.grad, graph=graph)

    def null_grad(self) -> "Tensor":
        """Sets this tensor's gradient to be ``None``.

        This operation is performed in-place, but a reference to the
        tensor is returned in order to permit mapping semantics.

        Returns
        -------
        self

        Examples
        --------
        >>> import  mygrad as mg
        >>> x = mg.Tensor(2.)
        >>> (x ** 2).backward()
        >>> x.grad
        array(4.)
        >>> x.null_grad()  # returns a reference of `x`
        Tensor(2.0)
        >>> x.grad is None
        True"""
        self.grad = None
        return self

    def null_gradients(self, clear_graph=True):
        """
        **Deprecated: Tensors will automatically have their computational graphs cleared during backprop.
        Simply involving a tensor in a new computational graph will null its gradient.**

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
        import warnings

        warnings.warn(
            "`tensor.null_gradients()` is deprecated. Calling it will raise an error "
            "in future versions of MyGrad. A tensor will automatically "
            "have its gradient nulled if you use it in a new computational graph. "
            "Or, you can call `tensor.null_grad()` to null that individual tensor's "
            "gradient.",
            FutureWarning,
        )

    def clear_graph(self):
        """
        Removes the current tensor – and tensors above it – from their shared
        computational graph.

        This de-references all operations involved in the graph and the intermediate
        tensors that were created by it.
        """
        self._restore_writeability()
        self._ops.clear()

        if self.creator is None:
            return

        creator = self._creator
        self._creator = None  # marks tensor as "visited" during graph-traversal

        for var in creator.variables:  # type: Tensor
            var.clear_graph()

    @property
    def scalar_only(self) -> bool:
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
    def constant(self) -> bool:
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
        return self._op(GetItem, self, op_args=(item,))

    def __iter__(self):
        # In the same way that numpy doesn't let you iterate over 0-dimensional
        # arrays, don't allow iteration over 0-dimensional arrays.
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return iter(self[n] for n in range(len(self)))

    def _mirror_tensor(self, tensor: "Tensor"):
        """ *Dev use only*

        Points all of the attributes of ``self`` to those of
        ``tensor`` so that they reference all of the same data structures.
        This is used to facilitate "in-place" operations.
        """
        self.__dict__ = tensor.__dict__

    def _make_placeholder_tensor(
        self, *, copy_data: bool, base: Optional["Tensor"] = None
    ) -> "Tensor":
        """
        Creates a tensor that stands in the place of `self` in the computational graph.

        The resulting tensor should never be exposed to the user; it is used to accommodate
        in-place operations.

        Parameters
        ----------
        copy_data : bool
            If True the placeholder holds a copy of the original data

        base : Optional[Tensor]
            Points the placeholder to the base tensor.

        Returns
        -------
        placeholder : Tensor
        """
        assert (
            self.grad is None
        ), "A placeholder copy can not be created for a tensor with a gradient"

        assert (
            not self._accum_ops
        ), "A placeholder copy cannot be created during backprop"

        placeholder = Tensor(
            self,
            constant=self.constant,
            _scalar_only=self._scalar_only,
            _creator=self.creator,
            _base=base,
            _copy_data=copy_data,
        )
        placeholder._ops = self._ops

        # point all ops involving `self` to old_tensor instead
        for op in placeholder._ops:
            op.variables = tuple(
                var_ if var_ is not self else placeholder for var_ in op.variables
            )

        return placeholder

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
        # TODO: make sure that a failed in-place op doesn't corrupt the graph
        # old_tensor is the tensor pre-setitem
        old_tensor = self._make_placeholder_tensor(copy_data=True, base=self.base)

        # self becomes the tensor post-setitem
        out = self._op(
            inplace_op,
            old_tensor,
            *input_vars,
            op_args=op_args,
            op_kwargs=op_kwargs,
            constant=constant,
        )
        if out.base is old_tensor:
            out._base = None
        self._mirror_tensor(out)

    def __setitem__(self, key, value):
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
        if isinstance(other, Number) or (
            isinstance(other, np.ndarray) and other.ndim == 0
        ):
            if other == 1:
                return self._op(Positive, self)
            elif other == 2:
                return self._op(Square, self)

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
        return self.copy()

    def copy(self, constant: Optional[bool] = None) -> "Tensor":
        """ Produces a copy of ``self`` with ``copy.creator=None``.

        Copies of the underlying numpy data array and gradient array are created.

        No information regarding the tensor's participation in the computational
        graph are copied.

        Parameters
        ----------
        constant : Optional[bool]

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
        copy = Tensor(
            np.copy(self.data),
            constant=(self.constant if constant is None else constant),
        )
        copy.grad = np.copy(self.grad) if self.grad is not None else None
        return copy

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
    def base(self) -> Optional["Tensor"]:
        """
        A reference to the base tensor if memory is from other tensor

         Examples
        --------
        The base of a tensor that owns its memory is ``None``:

        >>> import mygrad as mg
        >>> x = mg.arange(5)
        >>> x.base is None
        True

        Slicing creates a view, whose memory is shared with x:

        >>> y = x[2:]
        >>> y.base is x
        True
        >>> y.data.base is x.data
        True

        A view of a view has the same base as its "parent"

        >>> z = y[:]
        >>> z.base is x
        True

        The behavior of ``Tensor.base`` departs from that of ``ndarray.base`` in that
        mygrad will never create an "internal" tensor to serve as a base; e.g.

        >>> import numpy as np
        >>> np.reshape(2., (1,)).base
        array(2.)

        >>> mg.reshape(2., (1,)).base is None
        True
        """
        return self._base

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

    def __eq__(self, other):
        return np.ndarray.__eq__(
            self.data, other.data if isinstance(other, Tensor) else other
        )

    def __ne__(self, other):
        return np.ndarray.__ne__(
            self.data, other.data if isinstance(other, Tensor) else other
        )

    def __imatmul__(self, other):
        raise TypeError(
            "In-place matrix multiplication is not (yet) supported. "
            "Use 'a = a @ b' instead of 'a @= b'"
        )

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.data, dtype)


# set all comparison operators - mirrors ndarray methods
def tensor_to_array_wrapper(func):
    @wraps(func)
    def wrapped(x, y):
        return func(x.data, y.data if isinstance(y, Tensor) else y)

    return wrapped


for _op in ("__lt__", "__le__", "__gt__", "__ge__"):
    setattr(Tensor, _op, tensor_to_array_wrapper(getattr(np.ndarray, _op)))
