"""
This module defines the base tensor class along with all of its essential
attributes and special methods. Public math methods, e.g. ``sum``, ``mean``,
etc., are bound to the Tensor class in ``mygrad.__init__.py``.
"""

from numbers import Integral, Number
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from weakref import ReferenceType, finalize

import numpy as np

import mygrad._utils.duplicating_graph as _dup
import mygrad._utils.graph_tracking as _track
import mygrad._utils.lock_management as _mem
from mygrad._tensor_core_ops.indexing import GetItem, SetItem
from mygrad._utils import (
    WeakRef,
    WeakRefIterable,
    collect_all_operations_and_clear_grads,
)
from mygrad.errors import DisconnectedView
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
from mygrad.math.sequential.ops import (
    CumProd,
    CumSum,
    Max,
    Mean,
    Min,
    Prod,
    StdDev,
    Sum,
    Variance,
)
from mygrad.operation_base import Operation, _NoValue
from mygrad.tensor_manip.array_shape.ops import Flatten, Ravel, Reshape, Squeeze
from mygrad.tensor_manip.transpose_like.ops import (
    MoveAxis,
    SwapAxes,
    Tensor_Transpose_Property,
    Transpose,
)
from mygrad.typing import ArrayLike, DTypeLike, DTypeLikeReals, Index, Shape

__all__ = ["Tensor", "asarray", "astensor"]

CONSTANT_ONLY_DTYPES = (np.integer, np.bool_)


def _resolve_constant(*others: Any, constant: Optional[bool]) -> Optional[bool]:
    """Determines if `constant` should be resolved to True based on `others`.
    Otherwise defers to a tensor-creator to handle further resolutions based on dtype."""
    if constant is not None:
        return constant
    for other in others:
        if isinstance(other, Tensor) and not other.constant:
            # let subsequent tensor casting infer constant from dtype
            return None
    # all inputs are constants
    return True


def asarray(a: ArrayLike, dtype: DTypeLike = None, order: str = None) -> np.ndarray:
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
    if isinstance(a, Tensor):
        a = a.data  # faster than passing the tensor directly
    return np.asarray(a, dtype=dtype, order=order)


def tensor(
    arr_like: ArrayLike,
    dtype: DTypeLikeReals = None,
    *,
    constant: Optional[bool] = None,
    copy: bool = True,
    ndmin: int = 0,
) -> "Tensor":
    """
    Create a tensor

    This documentation was adapted from that of ``numpy.array`

    Parameters
    ----------
    arr_like : array_like
        A tensor, any object exposing the array interface, an object whose
        __array__ method returns an tensor, a real number, any (nested) sequence.

    dtype : data-type, optional
        The desired data-type for the tensor. Restricted to integer and float type.
        If not specified, then the type will be determined as the minimum type required
        to hold the objects in the sequence.

    constant : Optional[bool]
        If ``True``, this tensor is treated as a constant, and thus does not
        facilitate back propagation (i.e. ``constant_tensor.grad`` will always
        return ``None``).

        If a new tensor is returned:
         - Defaults to ``False`` for float-type data.
         - Defaults to ``True`` for integer-type data.

    copy : bool, optional
        If true (default), or if a copy is needed to satisfy any of the
        other requirements (``dtype``, ``constant``, etc.) then a new tensor
        is created from copied data. Otherwise the tensor will be returned
        unchanged.

    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        tensor should have. Ones will be pre-pended to the shape as
        needed to meet this requirement.

    Returns
    -------
    out : Tensor
        A tensor satisfying the specified requirements.

    See Also
    --------
    empty_like : Return an empty tensor with shape and type of input.
    ones_like : Return an tensor of ones with shape and type of input.
    zeros_like : Return an tensor of zeros with shape and type of input.
    full_like : Return a new tensor with shape of input filled with value.
    empty : Return a new uninitialized tensor.
    ones : Return a new tensor setting values to one.
    zeros : Return a new tensor setting values to zero.
    full : Return a new tensor of given shape filled with value.

    Examples
    --------
    >>> import mygrad as mg
    >>> mg.tensor([1, 2, 3])
    Tensor([1, 2, 3])

    Upcasting:

    >>> mg.tensor([1, 2, 3.0])
    Tensor([ 1.,  2.,  3.])

    More than one dimension:

    >>> mg.tensor([[1, 2], [3, 4]])
    Tensor([[1, 2],
            [3, 4]])

    Minimum dimensions 2:

    >>> mg.tensor([1, 2, 3], ndmin=2)
    Tensor([[1, 2, 3]])

    Type provided:

    >>> mg.tensor([1, 2, 3], dtype="float32")
    Tensor([1., 2., 3.], dtype=float32)
    """

    if isinstance(arr_like, Tensor) and copy is False:
        if (constant is None or arr_like.constant is constant) and (
            dtype is None or (arr_like.dtype == np.dtype(dtype))
        ):
            if not isinstance(ndmin, Integral):
                raise TypeError(
                    f"TypeError: `ndmin` requires a non-negative integer (got type {type(ndmin)})"
                )
            if ndmin < 0:
                ndmin = 0  # numpy does this
            if ndmin > arr_like.ndim:
                arr_like = arr_like[(*(None for _ in range(ndmin - arr_like.ndim)),)]
            # return tensor as-as
            return arr_like

    return Tensor(arr_like, dtype=dtype, constant=constant, copy=copy, ndmin=ndmin)


def astensor(
    t: ArrayLike, dtype: DTypeLikeReals = None, *, constant: Optional[bool] = None
) -> "Tensor":
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

        Defaults to ``False`` for float-type data.
        Defaults to ``True`` for integer-type data.

        Integer-type tensors must be constant.

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

    >>> a = np.array([1.0, 2.0])
    >>> mg.astensor(a)
    Tensor([1., 2.])
    >>> a is mg.astensor(a).data
    True

    Existing tensors are not copied and their gradients and
    computational graphs are preserved:

    >>> t1 = 2 * mg.tensor([1, 2])
    >>> t2 = mg.astensor(t1)
    >>> t1 is t2
    True
    >>> t1.creator is t2.creator
    True

    If `dtype` is set, a new tensor is created - with copied data - only
    if dtype does not match:

    >>> t = mg.Tensor([1, 2], dtype=np.float32)
    >>> mg.astensor(t, dtype=np.float32) is t
    True
    >>> mg.astensor(t, dtype=np.float64) is t
    False

    Otherwise, if `constant` is set, a new tensor is created (with
    no copy of the underlying data) only if constant doesn't match.

    >>> t1 = mg.tensor([1, 2], constant=False)
    >>> mg.astensor(t1, constant=False) is t
    True
    >>> mg.astensor(t1, constant=True) is t1
    False
    >>> mg.astensor(t1, constant=True).data is t1.data
    True
    """
    return tensor(t, dtype=dtype, constant=constant, copy=False, ndmin=0)


class Tensor:
    """A numpy-array-like object capable of serving as a node in a computational
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
    >>> mg.tensor(2.3)  # creating a 0-dimensional tensor
    Tensor(2.3)
    >>> mg.tensor(np.array([1.2, 3.0]))  # casting a numpy-array to a tensor
    Tensor([1.2, 3.0])
    >>> mg.tensor([[1, 2], [3, 4]])  # creating a 2-dimensional tensor
    Tensor([[1, 2],
            [3, 4]])
    >>> mg.arange(4)    # using numpy-style tensor creation functions
    Tensor([0, 1, 2, 3])

    Creating a non-constant tensor will copy array data:

    >>> import numpy as np
    >>> arr = np.arange(10.)
    >>> t_var = tensor(arr, constant=False)
    >>> np.shares_memory(arr, t_var)
    False

    Creating constant tensor will not make a copy of the array data:

    >>> t_const = mg.tensor(arr, constant=True)
    >>> np.shares_memory(arr, t_const)
    True

    Forward and Back-Propagation
    ----------------------------
    Let's construct a computational graph consisting of two zero-dimensional
    tensors, ``x`` and ``y``, which are used to compute an output tensor,
    ````. This is a "forward pass imperative" style for creating a computational
    graph - the graph is constructed as we carry out the forward-pass computation.

    >>> x = mg.tensor(3.0)
    >>> y = mg.tensor(2.0)
    >>> ℒ = 2 * x + y ** 2

    Invoking ``ℒ.backward()`` signals the computational graph to
    compute the total-derivative of ``f`` with respect to each one of its dependent
    variables. I.e. ``x.grad`` will store ``dℒ/dx`` and ``y.grad`` will store
    ``dℒ/dy``. Thus we have back-propagated a gradient from ``f`` through our graph.

    Each tensor of derivatives is computed elementwise. That is, if `x = Tensor(x0, x1, x2)`,
    then dℒ/dx represents `[dℒ/d(x0), dℒ/d(x1), dℒ/d(x2)]`

    >>> ℒ.backward()  # computes df/dx and df/dy
    >>> x.grad  # df/dx
    array(6.0)
    >>> y.grad  # df/dy
    array(4.0)
    >>> ℒ.grad
    array(1.0)  # dℒ/dℒ

    Once the gradients are computed, the computational graph containing ``x``,
    ``y``, and ``ℒ`` is cleared automatically. Additionally, involving any
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

    >>> x = mg.tensor([1, 2])
    >>> x.data
    array([1, 2])

    **Do not modify this underlying array**. Any in-place modifications made to this
    array will not be tracked by any computational graph involving that tensor, thus
    back-propagation through that tensor will likely be incorrect.

    Producing a "View" of a Tensor
    ------------------------------
    MyGrad's tensors exhibit the same view semantics and memory-sharing relationships
    as NumPy arrays. I.e. any (non-scalar) tensor produced via basic indexing will share
    memory with its parent.

    >>> x = mg.tensor([1., 2., 3., 4.])
    >>> y = x[:2]  # the view: Tensor([1., 2.])
    >>> y.base is x
    True
    >>> np.shares_memory(x, y)
    True

    Mutating shared data will propagate through views:

    >>> y *= -1
    >>> x
    Tensor([-1., -2.,  3.,  4.])
    >>> y
    Tensor([-1., -2.])

    And this view relationship will also manifest between the tensors' gradients

    >>> (x ** 2).backward()
    >>> x.grad
    array([-2., -4.,  6.,  8.])
    >>> y.grad
    array([-2., -4.])

    In-Place Operations are not Efficient
    =====================================
    It is important to note that while MyGrad's view semantics promote a rich parity
    with NumPy, that certain aspects should be avoided in the interest of optimized performance.
    Namely, performing in-place operations on tensors is generally not more efficient than
    their non-mutating counterparts.

    This is because MyGrad has to track the state of tensors that are involved in a computational
    graph. Thus a mutated tensor must have its pre-augmented state stored for future reference; this
    defeats the performance benefit of writing to an array's memory in-place. This is especially
    inefficient if you are mutating a tensor involved with multiple views of the same memory(
    By contrast, producing a view of a tensor _is_ efficient as one would expect).

    Thus these NumPy-like in-place semantics are supported by MyGrad not for the same performance
    purposes, but instead to support convenient and familiar code-patterns and to enable one to
    port NumPy code to MyGrad (or, in the future, inject MyGrad tensors into NumPy!!) and get
    the exact same behavior.

    A final note: MyGrad's in-place operations, when run under :func:`~mygrad.no_autodiff` mode,
    do not incur the extra costs noted above, and thus your code will benefit from the performance
    benefits of in-place operations.
    """

    __array_priority__ = 15.0

    def __array__(self, dtype: DTypeLike = None) -> np.ndarray:
        return np.array(self.data, dtype=dtype, copy=False)

    def __init__(
        self,
        x: ArrayLike,
        *,
        dtype: DTypeLikeReals = None,
        constant: Optional[bool] = None,
        copy: bool = True,
        ndmin: int = 0,
        _creator: Optional[Operation] = None,
        _base: Optional["Tensor"] = None,
    ):
        """
        Parameters
        ----------
        x : ArrayLike
            Input data, in any form that can be converted to an array.  This
            includes numbers, sequences, nested sequences, numpy-ndarrays,
            and mygrad-tensors.

        dtype : DTypeLikeReals
            `int`, `float`, or a real-valued numpy data type. By default the
            data type is inferred from ``x`` via ``numpy.asarray(x)``.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. `self.grad` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        copy : Optional[bool]
            Determines if the incoming array-data will be copied.

        ndmin : int, optional
            Specifies the minimum number of dimensions that the resulting
            array should have.  Ones will be prepended to the shape as
            needed to meet this requirement.

        Notes
        -----
        The following are parameters reserved only for internal use:

        _creator : Optional[mygrad.Operation]
            The operation-instance whose forward pass produced `self`. Should not
            be set manually by users.

        _base : Optional[Tensor]
            Points to the tensor that ``self`` shares memory with.
        """

        if constant is not None and not isinstance(constant, bool):
            raise TypeError(f"`constant` must be a boolean value, got: {constant}")

        self._creator: Optional[Operation] = _creator

        self.data = np.array(x, dtype=dtype, copy=copy, ndmin=ndmin)  # type: np.ndarray

        dtype = self.data.dtype.type
        is_float = issubclass(dtype, np.floating)  # faster than `numpy.issubdtype`
        if not is_float and _track.TRACK_GRAPH:
            # No need to constrain dtypes if we aren't tracking the graph.
            # Also, it is nice to enable complex arithmetic through mygrad
            # functions that are wrapped in no_autodiff
            if not issubclass(dtype, CONSTANT_ONLY_DTYPES):
                raise TypeError(
                    f"Tensor data must be of an floating type, integer type, or boolean type, "
                    f"received {dtype}"
                )

            elif constant is False:
                raise ValueError("Integer-valued tensors must be treated as constants.")

        if constant is None:
            # non-float: default constant -> True
            # float: default constant -> False
            constant = not is_float

        self._constant = constant

        self._grad = None  # type: Union[None, np.ndarray]

        # track all operations that this tensor participates in
        self._ops = set()  # type: Set[WeakRef[Operation]]

        # track the operations that have contributed to this tensor's gradient during a back-prop
        self._accum_ops = set()  # type: Set[WeakRef[Operation]]

        # base points to the initial tensor that owns the memory of this
        # tensor
        self._base = _base  # type: Optional[Tensor]
        # stores all of the tensors that are a view of this tensor
        self._view_children = WeakRefIterable()  # type: WeakRefIterable[Tensor]

        # Used to reflect the view of the gradient associated with that of `self.base`.
        # This is a means of distinguishing between the gradient set on `self` as
        # part of backpropagation and the view of the gradient of its base.
        self._view_grad: Optional[np.ndarray] = None

    @property
    def grad(self) -> Optional[np.ndarray]:
        """
        Returns the derivative of ``ℒ`` with respect to this tensor.

        ``ℒ`` is the terminal node in the compuational graph from which
        ``ℒ.backward()`` was invoked.

        If this tensor is a view of another tensor then their gradients
        will exhibit the same memory-sharing relationship as their data.

        Returns
        -------
        dℒ/dx: numpy.ndarray
            The gradient of the terminal node in a computational graph
            with respect to this tensor. The shape of this numpy array
            matches ``self.shape``

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.Tensor([1.0, 2.0])

        Prior to backpropagation tensors have ``None`` set for their gradients.

        >>> x.grad is None
        True

        Now we trigger backpropagation...

        >>> ℒ = x ** 2
        >>> ℒ.backward()

        and we see that ``x.grad`` stores dℒ/dx

        >>> x.grad  # dℒ/dx
        array([2., 4.])

        Now we will demonstrate the relationship between gradient a view tensor
        and that of its base.

        >>> base = mg.Tensor([1.0, 2.0, 3.0])
        >>> view = base[:2]; view
        Tensor([1., 2.])

        >>> ℒ = base ** 2
        >>> ℒ.backward()

        Although ``view`` is not directly involved in the computation in ``ℒ``,
        and thus would not typically store a gradient in due to ``ℒ.backward()``,
        it shares memory with ``base`` and thus it stores a gradient in correspondence
        to this "view relationship". I.e. because ``view == base[:2]``, then we expect
        to find that ``view.grad == base.grad[:2]``.

        >>> base.grad
        array([2., 4., 6.])
        >>> view.grad
        array([2., 4.])

        >>> view.grad.base is base.grad
        True

        The reasoning here is that, because a base tensor and its view share the same
        array data, then varying an element in that data implies that both the base
        tensor and the view will change (assuming the variation occurs specifically in
        a shared region). It follows that the base tensor's gradient must share the same
        relationship with the view-tensor since these are measures of "cause and effects"
        associated with varying elements of data (albeit infinitesmaly).
        """
        if self._base is None:
            return self._grad

        if self._view_grad is not None and self._view_grad.base is self._base._grad:
            # view grad has been computed already
            return self._view_grad

        if self._base._grad is None or self._creator is None:
            #  ``self`` had its graph, connecting it to its base, cleared.
            #  ``self._view_grad`` can't be computed without this info.
            return None

        (view_parent,) = self._creator.variables

        # recursively fetches grad from parent
        grad = view_parent.grad
        with _track.no_autodiff:
            self._view_grad = self._replay_op(grad).data if grad is not None else None
        return self._view_grad

    def astype(
        self, dtype: DTypeLikeReals, *, constant: Optional[bool] = None
    ) -> "Tensor":
        """Copy of the tensor with the specified dtype.

        The resulting tensor is not involved in any computational graph
        and has no gradient associated with it.

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
        return type(self)(self.data, dtype=dtype, copy=True, constant=constant)

    @classmethod
    def _op(
        cls,
        Op: Type[Operation],
        *input_vars: ArrayLike,
        op_args: Optional[Sequence] = None,
        op_kwargs: Optional[Dict[str, Any]] = None,
        constant: bool = None,
        out: Optional[Union[np.ndarray, "Tensor"]] = None,
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

        out: Optional[Union[np.ndarray, "Tensor"]]
            The target where the output (an ndarray) of the operation will be written.
            Thus this raises if `out` is read-only.

            There is an exception to this if  a tensor is provided, in which case the
            operation does not write to its underlying memory but rather triggers "in-place
            semantics" so that the computational graph behaves as if the tensor was mutated.
            See  ``Tensor._in_place_op`` for more details.

        Returns
        -------
        mygrad.Tensor
            The tensor-result of the operation's forward-pass."""
        if out is not None and isinstance(out, Tensor):
            out._in_place_op(
                Op, *input_vars, op_args=op_args, op_kwargs=op_kwargs, constant=constant
            )
            return out

        _uniques_bases_then_arrs = ()

        tensor_vars = tuple(
            cls(var, constant=True, copy=False) if not isinstance(var, Tensor) else var
            for var in input_vars
        )

        # cast all input-vars to tensors
        if _track.TRACK_GRAPH and _mem.MEM_GUARD:
            # lock memory of array data
            _uniques_bases_then_arrs = WeakRefIterable(
                _mem.lock_arr_writeability(x)
                for x in _mem.unique_arrs_and_bases(tensor_vars)
            )

        if op_args is None:
            op_args = tuple()

        if op_kwargs is None:
            op_kwargs = {}

        f = Op()

        try:
            if out is None:
                op_out: np.ndarray = f(*tensor_vars, *op_args, **op_kwargs)
            else:
                op_out: np.ndarray = f(*tensor_vars, *op_args, **op_kwargs, out=out)
        except Exception as e:
            if _track.TRACK_GRAPH and _mem.MEM_GUARD:
                _mem.release_writeability_lock_on_op(_uniques_bases_then_arrs)
            raise e

        if not _track.TRACK_GRAPH:
            # execute operation without tracking creator or any graph
            # information
            return cls(
                op_out,
                constant=constant,  # constant not determined by graph info
                copy=False,
                _creator=None,
                _base=None,
            )

        # points to parent tensor that op-output is a view of
        base = None  # type: Optional[Tensor]

        # If output of op is a view - tracks the tensor var that is
        # the parent of the view
        parent_var: Optional[Tensor] = None

        # Determine whether or not op was a view; if so, `base`
        # points to parent Tensor
        op_out_base = op_out.base
        if f.can_return_view and op_out_base is not None:
            vars_can_share_mem = (
                isinstance(var, (np.ndarray, Tensor)) for var in input_vars
            )
            for can_share_mem, parent_var in zip(vars_can_share_mem, tensor_vars):
                if not can_share_mem:
                    continue
                parent_data = parent_var.data
                parent_data_base = parent_data.base

                if (
                    (op_out_base is parent_data)
                    or (op_out_base is parent_data_base)
                    or (op_out is parent_data)
                ):
                    if parent_var._base is not None and parent_var._creator is None:
                        parent_var._base = None

                    base = parent_var if parent_var.base is None else parent_var.base
                    break
            else:
                parent_var = None

        for v in input_vars:
            if isinstance(v, Tensor):
                # tensor's graph has been cleared, but its base lingers
                if v._base is not None and v._creator is None:
                    v._base = None

                if base is None:
                    # non-view ops clear grads
                    v._grad = None
                    v._view_grad = None

        if base is not None:
            # we need to be able to replay view-ops for doing in-place operations
            # on graphs with views
            f.replay_args = op_args
            f.replay_kwargs = op_kwargs
            f.replay_force_constant = constant

        # record graph information
        if constant is None:
            if any(not var.constant for var in tensor_vars):
                constant = None
            else:
                constant = True

        # record that a variable participated in that op
        ref_f = ReferenceType(f)  # type: WeakRef[Operation]
        for var in tensor_vars:
            var._ops.add(ref_f)

        tensor_out = cls(
            op_out,
            constant=constant,
            copy=False,
            _creator=f,
            _base=base,
        )

        if parent_var is not None:
            parent_var._view_children.append(tensor_out)

        if _mem.MEM_GUARD:
            if out is not None and tensor_out.data.base is not None:
                _mem.lock_arr_writeability(tensor_out.data.base)
                _uniques_bases_then_arrs.append(tensor_out.data.base)
            _mem.lock_arr_writeability(tensor_out.data)
            tensor_refs = _uniques_bases_then_arrs
            tensor_refs.append(tensor_out.data)
            finalize(f, _mem.release_writeability_lock_on_op, tensor_refs)
        return tensor_out

    def _replay_op(self, *input_vars: ArrayLike) -> "Tensor":
        """*dev use only*

        Replays the op that produced `self` - called on the specified
        input vars"""
        if self.creator is None:
            raise DisconnectedView(
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

    def backward(self, grad: Optional[ArrayLike] = None):
        """Trigger backpropagation and compute the derivatives of this tensor.

        Designating this tensor as the tensor ℒ, compute dℒ/dx for all (non-constant) tensors
        that preceded ℒ in its computational graph, and store each of these derivatives in ``x.grad``
        respectively.

        Once back-propagation is finished, the present tensor is removed from all computational
        graphs, and the preceding graph is cleared.

        If ℒ is a non-scalar tensor (i.e. ``ℒ.ndim`` is greater than 0), then calling
        ``ℒ.backward()`` will behave as if ℒ was first reduced to a scalar via summation. I.e. it
        will behave identically to ``ℒ.sum().backward()``; this ensures that each element of any
        dℒ/dx will represent a derivative of a scalar function.

        Parameters
        ----------
        grad : Optional[array_like], (must be broadcast-compatible with ``self``
            By default, the present tensor is treated as the terminus of the computational graph (ℒ).
            Otherwise, one can specify a "downstream" derivative, representing ``dℒ/d(self)``.
            This can be used to effectively connect otherwise separate computational graphs.

        Examples
        --------
        >>> import mygrad as mg
        >>> x = mg.tensor(2)
        >>> y = mg.tensor(3)
        >>> w = x * y
        >>> ℒ = 2 * w
        >>> ℒ.backward()  # computes dℒ/dℒ, dℒ/dw, dℒ/dy, and dℒ/dx

        >>> ℒ.grad  # dℒ/df == 1 by identity
        array(1.)
        >>> w.grad  # dℒ/dw
        array(2.)
        >>> y.grad # dℒ/dy = dℒ/dw * dw/dy
        array(4.)
        >>> x.grad # dℒ/dx = dℒ/dw * dw/dx
        array(6.)

        Calling ``ℒ.backward()`` from a non-scalar tensor is equivalent
        to first summing that tensor.

        >>> tensor = mg.tensor([2.0, 4.0, 8.0])
        >>> ℒ = tensor * tensor[::-1]  # [x0*x2, x1*x1, x2*x0]
        >>> ℒ.backward()  # behaves like ℒ = x0*x2 + x1*x1 + x2*x0
        >>> tensor.grad
        array([16.,  8.,  4.])

        >>> tensor = mg.Tensor([2.0, 4.0, 8.0])
        >>> ℒ = tensor * tensor[::-1]
        >>> ℒ.sum().backward()
        >>> tensor.grad
        array([16.,  8.,  4.])

        Specifying a value for ``grad``

        >>> x = mg.Tensor(1.)
        >>> x.backward(2.)
        >>> x.grad  # Would normally be dℒ/dℒ == 1
        array(2.)
        """
        if not _track.TRACK_GRAPH:
            return

        if self.constant:
            self.clear_graph()
            return

        # don't set self._grad yet because there is a grad-clearing step that
        # occurs during graph creation
        if grad is not None:
            # `self` is guaranteed to be a tensor of floats
            # so we can simply cast `grad` to be the same dtype
            _grad = asarray(grad, dtype=self.dtype)

            if _grad.shape != self.shape:
                try:
                    # See if grad can broadcast to `self`
                    # raises ValueError if not
                    _grad = np.multiply(
                        np.full_like(self.data, fill_value=1.0),
                        _grad,
                        dtype=self.dtype,
                    )
                    if _grad.shape != self.shape:
                        # mutual broadcasting occurred
                        raise ValueError()
                except ValueError:
                    raise ValueError(
                        f"`tensor.backward(grad)` was passed a gradient with an incompatible shape.\n"
                        f"`grad` must be broadcast-compatible with `tensor.shape={self.shape}`\n"
                        f"Got `grad.shape={_grad.shape}`"
                    )
        else:
            _grad = np.full_like(self.data, fill_value=1.0)

        if self.creator is not None:
            # stores a set of all the operation-instances that participate in
            # the computational graph up to and including the present operation
            graph = set()  # type: Set[WeakRef[Operation]]

            # populates graph and clears all grads
            collect_all_operations_and_clear_grads(self, seen=graph)
            self._grad = _grad
            self._backward(graph=graph)
        else:
            self._grad = _grad

        self.clear_graph()

    def _backward(self, *, graph: Set[WeakRef[Operation]]):
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
        assert self._grad is not None, (
            f"backprop, post grad-accumulation, was triggered "
            f"on a tensor with no gradient"
            f"\n{self}"
            f"\nid {id(self._ops)}"
            f"\ngrad: {self.grad}"
            f"\ncreator: {self.creator}"
            f"\nops: {self._ops}"
            f"\nbase: {self.base}"
        )
        assert self._grad.shape == self.shape, (
            f"A tensor and its associated gradient must possess the same shape. Got:"
            f"\ntensor-shape: {self.shape}"
            f"\ngrad-shape: {self._grad.shape}"
        )
        self._ops.difference_update(self._accum_ops)
        self._accum_ops.clear()
        if self.creator is not None and self._ops.isdisjoint(graph):
            self._creator.backward(self._grad, graph=graph)

    def null_grad(self, *, _clear_view_info: bool = False) -> "Tensor":
        """Sets this tensor's gradient to be ``None``.

        This operation is performed in-place, but a reference to the
        tensor is returned in order to permit mapping semantics.

        Also removes any ``base`` reference from disconnected views.

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
        self._view_grad = None
        self._grad = None

        if _clear_view_info:
            if self._base is not None and self._creator is None:
                self._base = None

        return self

    def null_gradients(self, clear_graph: bool = True):
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
        >>> x = mg.tensor(2)
        >>> y = mg.tensor(3)
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
        tensors that were created by it. Arrays whose memory were locked by the
        computational graph will have their writeability restored.

        Examples
        --------
        >>> import mygrad as mg
        >>> import numpy as np
        >>> x = np.array([1., 2.])
        >>> y = mg.multiply(2., x)
        >>> x.flags.writeable, y.creator
        (False, <mygrad.math.arithmetic.ops.Multiply at 0x224f89cac48>)
        >>> y.clear_graph()
        >>> x.flags.writeable, y.creator
        (True, None)
        """
        if self._base is not None:
            # "pull" on grad to force views to update their
            # gradients from upstream before the graph info
            # gets cleared
            _ = self.grad

        self._view_children.clear()
        self._ops.clear()

        if self._creator is None:
            return

        creator = self._creator
        self._creator = None  # marks tensor as "visited" during graph-traversal

        for var in creator.variables:  # type: Tensor
            var.clear_graph()

    @property
    def constant(self) -> bool:
        """If ``True``, this tensor is a constant; it will not propagate any gradient.

        Additionally, any tensor that is a descendant of constant tensors will also
        be a constant.

        Integer-valued tesnors, Python scalars and NumPy arrays are treated as constant
        tensors when included in MyGrad computational graphs.

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

        Integer-valued tensors are treated as constants

        >>> mg.Tensor([1, 2]).constant
        True
        """
        return self._constant

    @property
    def creator(self) -> Operation:
        """The ``Operation`` instance that produced ``self``.

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

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, item) -> bool:
        return self.data.__contains__(item)

    def __getitem__(self, item: Index) -> "Tensor":
        return self._op(GetItem, self, op_args=(item,))

    def __iter__(self) -> Iterator["Tensor"]:
        # In the same way that numpy doesn't let you iterate over 0-dimensional
        # arrays, don't allow iteration over 0-dimensional arrays.
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return iter(self[n] for n in range(len(self)))

    def _in_place_op(
        self,
        inplace_op: Type[Operation],
        *input_vars: ArrayLike,
        op_args: Optional[Sequence] = None,
        op_kwargs: Optional[Dict] = None,
        constant: bool = None,
    ):
        if _track.TRACK_GRAPH is False:
            return self._op(
                inplace_op,
                *input_vars,
                op_args=op_args,
                op_kwargs=op_kwargs,
                constant=constant,
                out=self.data,
            )
        #
        # **********************************************************************************
        # The way that in-place updates work in MyGrad is that any tensor that
        # is about to undergo a mutation gets "cloned". Each resulting "placeholder"
        # is used to represent that tensor in any non-view operations that the tensor
        # was participating in. This ensures that the stateful computational graph
        # is not corrupted by this mutation.
        #
        # Once the placeholders have been created, they have permanently replaced the
        # rolls of their counterparts within the computational graph. Furthermore, they
        # exist only internally to the computational graph and thus cannot be the targets
        # of subsequent views or in-place updates.
        #
        # At this point, the "original" tensors merely reserve the publicly-available
        # Tensor-instances (husks) that the users will access. We eventually need to
        # populate these husks with the appropriate augmented contents and graph-history.
        #
        # Thus this method will compute the in-place operation on a new tensor, and
        # will create a new, internal computational graph involving the base tensor
        # affected by the mutation and any of its view-children. These tensors represent
        # the mutated tensors that the users expect to have access to.
        #
        # We must connect this new computational graph to the preceding one – the one
        # involving the placeholders; this way we can backpropagate appropriately and
        # through all influencers.
        #
        # Finally we mirror each of these new tensors into the husks of the publicly
        # -available tensors and reroute the computational graph through them so that
        # the user sees that all of the relevant tensors have been augmented, and that
        # they are connected to the appropriate "history" such that backprop occurs
        # without error or inaccuracy.
        #
        #
        # For illustration, consider the following graph:
        #
        # ... x------[square]-- y = x**2
        #        \
        #         ---[slice]-- z = view-x
        #                              \
        #                               ---[mul]-- w = 3 * z
        #
        # Now suppose that we mutate `x` with `x[:] = 0`. This is a simpler case than
        # mutating a view of `x`, since `x` is already the base tensor.
        #  - This should not affect `y`
        #  - It should affect `view_x`
        #  - It should *not* affect `w`, which depends on `view_x` in a "static" way.
        #    I.e. the value for `w` is already resolved and is not a view of z or x.
        #
        #
        # As prescribed above, we will make the placeholders: px and pz, and we
        # will reroute the operations that statically depend on the old values of x and z
        # through these placeholders.
        #
        # Next we will have `x` point to a mutated version of itself, in accord with the
        # in-place update being performed, and we will subsequently recreate any
        # views of x (i.e. z), based off of this mutated tensor.
        #
        # The resulting graph is:
        #
        #                             ---[slice]-- z = view-x
        #                            /
        #        -----[set-item] -- x = px.copy()[:]=0
        #       /
        # ... px------[square]-- y = px**2
        #        \
        #         ---[slice]-- pz = view-px
        #                              \
        #                               ---[mul]-- w = 3 * pz
        #
        # Note that px and pz are strictly *internal* tensors; they cannot be accessed for
        # use in any further operations, whereas `x` and `z` are available for further use.
        #
        # **********************************************************************************
        #
        # Replace base and all of its views with "placeholder" tensors;
        # they serve as internal references to all tensors pre-mutation
        # and will preserve ops relying on the un-mutated tensors.
        #
        # These placeholder tensors are never publicly-available and thus cannot
        # be involved directly in future in-place updates

        # In Tensor._op, any tensor entering an op has its grad/view-info cleared
        # We must do this here up front since we need to consume information
        # about ``self``
        self.null_grad(_clear_view_info=True)
        if self._base is not None and not self._base._view_children:
            self._base = None

        graph = _dup.DuplicatingGraph(self if self.base is None else self.base)

        # Create copy of base so that mutation has no impact on the
        # state of any ops depending on it or its views
        mutant_base = graph.base.tensor.copy()
        mutant_base.data.flags.writeable = (
            graph.base.tensor.data.flags.writeable
            or _mem.array_is_tracked(graph.base.tensor.data)
        )

        # Create view of base in correspondence to relationship
        # that `self` has to base. Mutating this view will mutate
        # base appropriately
        inplace_target = mutant_base

        # stores view-fn sequence from base -> in-place target
        view_fn_sequence: List[Callable[[np.ndarray], np.ndarray]] = []

        with _track.no_autodiff:
            # get view sequence from base -> in-place target
            for node in graph.get_path_to_base(self)[::-1][1:]:  # skip base
                # need to point to place-holder replay op to avoid creating
                # forwards references to downstream tensors
                f = node.placeholder._replay_op
                if self.base is not None:
                    # need sequence of view-ops
                    view_fn_sequence.append(_track.no_autodiff(f, to_numpy=True))
                inplace_target = f(inplace_target)

        # Constant info was not propagated through no-autodiff mode.
        # It must be inferred from the original tensor
        inplace_target._constant = mutant_base.constant

        mutant_base_data = mutant_base.data
        del mutant_base

        try:
            with _mem.mem_guard_off:
                placeholder_mutant_view = (
                    self._op(  # will raise if original data not writeable
                        inplace_op,
                        *(graph.get_placeholder_if_exists(t) for t in input_vars),
                        op_args=op_args,
                        op_kwargs=op_kwargs,
                        constant=constant,
                        out=inplace_target.data,
                    )
                )
        except Exception as e:
            graph.restore_old_graph()
            raise e

        if _mem.MEM_GUARD:
            _mem.force_lock_tensor_and_creators(placeholder_mutant_view)

        if placeholder_mutant_view.creator.where is not True:
            # An operation like `multiply(x, y, where=mask, out=z)` occurred.
            # `placeholder_mutant_view` is the mutated version of `z`.
            # We need to connect the upstream version of `z` to the computational
            # graph so that `~mask * dℒ/dz` backprops to it, whereas `~mask * dℒ/dz`
            # will backprop to `x` and `y`.
            #
            # This is basically an alternative to treating `multiply(x, y, where=mask, out=z)`
            # like a three-input operation, which adds complexity to the implementation of
            # every op that supports `where` and `out`.
            #
            #               old-z ---------------------
            #                 |                       |
            #   multiply(x, y, where=mask, out=z)     |
            #                 |                       |
            #                 z    --------------------
            #                 |    |
            #                 ApplyMask
            #                    |
            #                    z
            with _mem.mem_guard_off:
                placeholder_mutant_view = type(self)._op(
                    _dup.ApplyMask,
                    placeholder_mutant_view,  # gets passed through unchanged
                    # ~mask * grad  backprops to upstream placeholder
                    graph[self].placeholder,
                    op_kwargs=dict(
                        mask=placeholder_mutant_view.creator.where,
                    ),
                )

        # Connect public base tensor to placeholder graph via the mutated placeholder
        # tensor `out`.
        if self.base is None:
            # The current graph:
            #    base-p --> | inplace | --> vp'
            # Becomes:
            #    base-p --> | inplace | --> base'
            #
            # The base tensor itself was the target of the in-place operation,
            # thus we need simply mirror original base against the mutant placeholder.
            # This effectively connects the original base to the placeholder graph
            mutant_base = placeholder_mutant_view

        else:
            # in-place operation occurred on a view; must connect mutated base
            # to graph and then reproduce downstream views
            #
            # The current graph:
            #    vp --> | inplace | --> vp'
            #
            # Becomes:
            #
            #    vp --> | inplace | --> vp' --> |        |
            #                                   | unview | --> base'
            #   base-p -----------------------> |        |
            #
            # I.e. the mutated base is a combination of the placeholder
            # base and of the mutant view.

            mutant_base = type(self)._op(
                _dup.UnView,
                graph.base.placeholder,
                placeholder_mutant_view,
                op_kwargs=dict(
                    # Copy to avoid upstream placeholder mutant view sharing memory
                    # with downstream mutant base
                    mutant_base_data=mutant_base_data,
                    view_fn_sequence=view_fn_sequence,
                ),
            )

        del placeholder_mutant_view

        # The original base now points to the augmented array data
        # and has the InPlaceOp as its creator
        _dup.mirror_tensor(source=mutant_base, target=graph.base.tensor)

        del mutant_base

        # Now that the base-tensor has been incorporated into the graph,
        # recreate the view-graph and reroute all tensors from previous
        # graph to their downstream counterparts
        #
        # Note that iterating in a topologically-ordered way is critical
        # here: each parent is updated before creating one of its children
        #
        # Iteration is always based off of the placeholders' relative positions
        # in the graph since this will never be mutated.
        for node in graph:
            if node.parent is None:
                continue
            view = node.tensor._replay_op(node.parent)
            _dup.mirror_tensor(source=view, target=node.tensor)
            node.parent._view_children.append(node.tensor)

    @property
    def shape(self) -> Shape:
        """Tuple of tensor dimension-sizes.

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

        The shape attribute can also be set to reshape the tensor in-place

        >>> y.shape = (1, 6, 1)
        >>> y
        Tensor([[[1],
                 [2],
                 [3],
                 [4],
                 [5],
                 [6]]])

        See Also
        --------
        mygrad.reshape : similar function
        Tensor.reshape : similar method"""
        return self.data.shape

    @shape.setter
    def shape(self, newshape: Union[int, Shape]):
        # Even though this op cannot mutate views, we still must
        # do graph-replaying here so that views can still reference
        # this tensor, but with the proper reshaping mediating them.
        #
        # E.g.
        # x = arange(10)   # shape-(10,)
        # y = x[:6]        # shape-(6,)
        # x.shape = (2, 5) # shape-(2, 5)
        #
        # y.base points to the shape-(2,5) array
        # even though y is a view of the flat array
        #
        # thus we need to play this graph as
        #   (history)
        #       |
        #   placeholder   shape-(10,)
        #       |-reshape
        #       x         shape-(2,5)
        #       |-reshape
        #   placeholder   shape-(10,)
        #       |-getitem
        #       y         shape-(4,)

        if not _track.TRACK_GRAPH:
            self.data.shape = newshape
            return

        if newshape == self.shape:
            return

        old_shape = self.shape

        # raise here if the shape is not compatible
        self.data.shape = newshape
        self.data.shape = old_shape

        # create placeholders for self and all of its view-children
        graph = _dup.DuplicatingGraph(self)
        # need to iterate over all nodes now before we tinker
        # with the view children
        nodes = tuple(graph)

        # reshape placeholder of self
        out = graph.base.placeholder.reshape(newshape)

        # Store contents of `out` in `self` and replace `out` in
        # graph with `self`
        out._base = graph.base.placeholder.base
        _dup.mirror_tensor(source=out, target=self)
        _dup.reroute_ops_through(source=out, target=self)
        del out

        # although `self` is a view of placeholder, placeholder
        # is stricly an internal tensor, we won't expose it as
        # base
        graph.base.placeholder._view_children.append(self)
        base = graph.base.placeholder.base

        if base is not None:
            # if `self` was a view, we need to update that parent's
            # view children so that it points to the placeholder
            creator = graph.base.placeholder.creator.variables[0]
            creator._view_children = WeakRefIterable(
                [
                    w if w is not self else graph.base.placeholder
                    for w in graph.base.placeholder._view_children
                ]
            )

        # Undo the reshape, and place this as the tensor joining
        # the reshaped `self` with the views of unshaped `self`
        unshaped = self.reshape(old_shape)

        for node in nodes:
            if node.parent is None:
                continue
            # direct what would be views of `self` to be views of `unshaped`,
            # which translates the mutated shape of `self` to the original
            # shape used to create the views
            parent = node.parent if node.parent is not self else unshaped
            view = node.tensor._replay_op(parent)
            _dup.mirror_tensor(source=view, target=node.tensor)
            _dup.reroute_ops_through(source=view, target=node.tensor)
            parent._view_children.append(node.tensor)

    def __setitem__(self, key: Index, value: ArrayLike):
        self._in_place_op(SetItem, self, value, op_args=(key,))

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self._op(Add, self, other)

    def __iadd__(self, other: ArrayLike) -> "Tensor":
        self._in_place_op(Add, self, other)
        return self

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self._op(Add, other, self)

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self._op(Subtract, self, other)

    def __isub__(self, other: ArrayLike) -> "Tensor":
        self._in_place_op(Subtract, self, other)
        return self

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._op(Subtract, other, self)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self._op(Divide, self, other)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._op(Divide, other, self)

    def __floordiv__(self, other: ArrayLike) -> "Tensor":
        if not self.constant:
            raise ValueError(
                "Floor division cannot involve non-constant mygrad tensors."
            )
        if isinstance(other, Tensor):
            other = other.data
        return type(self)(self.data.__floordiv__(other), constant=True)

    def __rfloordiv__(self, other: ArrayLike) -> "Tensor":
        if not self.constant:
            raise ValueError(
                "Floor division cannot involve non-constant mygrad tensors."
            )
        return type(self)(self.data.__rfloordiv__(other), constant=True)

    def __itruediv__(self, other: ArrayLike) -> "Tensor":
        self._in_place_op(Divide, self, other)
        return self

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self._op(Multiply, self, other)

    def __imul__(self, other: ArrayLike) -> "Tensor":
        self._in_place_op(Multiply, self, other)
        return self

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self._op(Multiply, other, self)

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        return self._op(MatMul, self, other)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return self._op(MatMul, other, self)

    def __pow__(self, other: ArrayLike):
        if isinstance(other, Number) or (
            isinstance(other, np.ndarray) and other.ndim == 0
        ):
            if other == 1:
                return self._op(Positive, self)
            elif other == 2:
                return self._op(Square, self)

        return self._op(Power, self, other)

    def __ipow__(self, other: ArrayLike) -> "Tensor":
        if isinstance(other, Number) or (
            isinstance(other, np.ndarray) and other.ndim == 0
        ):
            if other == 1:
                self._in_place_op(Positive, self)
                return self
            elif other == 2:
                self._in_place_op(Square, self)
                return self

        self._in_place_op(Power, self, other)
        return self

    def __rpow__(self, other: ArrayLike):
        return self._op(Power, other, self)

    def __neg__(self):
        return self._op(Negative, self)

    def __pos__(self):
        return self._op(Positive, self)

    def __repr__(self) -> str:
        return repr(self.data).replace("array", "Tensor").replace("\n", "\n ")

    def __copy__(self) -> "Tensor":
        """Produces a copy of ``self`` with ``copy.creator=None``.

        Copies of the underlying numpy data array and gradient array are created.

        Returns
        -------
        Tensor
        """
        return self.copy()

    def copy(self, *, constant: Optional[bool] = None) -> "Tensor":
        """Produces a copy of ``self`` with ``copy.creator=None``.

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
        copy._grad = np.copy(self._grad) if self._grad is not None else None
        return copy

    def item(self) -> Union[int, float]:
        """Copy an element of a tensor to a standard Python scalar and return it.

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
        float"""
        if self.size > 1:
            raise ValueError("can only convert a tensor of size 1 to a Python scalar")
        return self.data.item()

    def __float__(self) -> float:
        if self.size > 1:
            raise TypeError("can only convert a tensor of size 1 to a Python scalar")
        return float(self.data)

    def __int__(self) -> int:
        if self.size > 1:
            raise TypeError("can only convert a tensor of size 1 to a Python scalar")
        return int(self.data)

    def flatten(self, *, constant: bool = None) -> "Tensor":
        """Return a copy of the tensor collapsed into one dimension.

        This docstring was adapted from ``numpy.ndarray.flatten``.

        Parameters
        ----------
        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        Returns
        -------
        mygrad.Tensor
            A copy of the input tensor, flattened to one dimension.

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
        A reference to the base tensor that the present tensor is a view of.

        It this tensor owns its memory, then this returns ``None``.

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
    def size(self) -> int:
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
    def ndim(self) -> int:
        """Number of tensor dimensions. I.e. the number
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
        Tensor(1)

        >>> y = mg.Tensor([[1, 2, 3],
        ...                [4, 5, 6]])
        >>> y.ndim
        2
        >>> y[0, 0]  # two indices are required to identify an element in `x`
        Tensor(1)"""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the tensor's elements.

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

    def reshape(self, *newshape: Union[int, Shape], constant: bool = None) -> "Tensor":
        """Returns a tensor with a new shape, without changing its data.
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
    def T(self) -> "Tensor":
        """Same as self.transpose(), except that self is returned if self.ndim < 2 and
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

    def __eq__(self, other: ArrayLike) -> np.ndarray:
        return np.ndarray.__eq__(self.data, asarray(other))

    def __ne__(self, other: ArrayLike) -> np.ndarray:
        return np.ndarray.__ne__(self.data, asarray(other))

    def __lt__(self, other: ArrayLike) -> np.ndarray:
        return np.ndarray.__lt__(self.data, asarray(other))

    def __le__(self, other: ArrayLike) -> np.ndarray:
        return np.ndarray.__le__(self.data, asarray(other))

    def __gt__(self, other: ArrayLike) -> np.ndarray:
        return np.ndarray.__gt__(self.data, asarray(other))

    def __ge__(self, other: ArrayLike) -> np.ndarray:
        return np.ndarray.__ge__(self.data, asarray(other))

    def __imatmul__(self, other):
        raise TypeError(
            "In-place matrix multiplication is not (yet) supported. "
            "Use 'a = a @ b' instead of 'a @= b'"
        )

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Sum of tensor elements over a given axis.

        Parameters
        ----------
        axis : Optional[int, Tuple[ints, ...]]
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

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        sum_along_axis : mygrad.Tensor
            A Tensor with the same shape as `self`, with the specified
            axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
            a 0-dim Tensor is returned.

        See Also
        --------
        mygrad.Tensor.sum : Equivalent method.

        cumsum : Cumulative sum of array elements.

        mean, average

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        The sum of an empty tensor is the neutral element 0:

        >>> mygrad.sum([])
        Tensor(0.0)

        Examples
        --------
        >>> import mygrad as mg
        >>> import numpy as np
        >>> x = mg.tensor([1., 1.])
        >>> x.sum()
        Tensor(2.0)
        >>> x = mg.tensor([0.5, 0.7, 0.2, 1.5])
        >>> x.sum(dtype=np.int32)
        Tensor(1)
        >>> x = mg.tensor([[0, 1], [0, 5]])
        >>> x.sum()
        Tensor(6)
        >>> x.sum(axis=0)
        Tensor([0, 6])
        >>> x.sum(axis=1)
        Tensor([1, 5])
        """
        return Tensor._op(
            Sum, self, op_kwargs=dict(axis=axis, keepdims=keepdims), constant=constant
        )

    def prod(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Return the product of array elements over given axes.

        Parameters
        ----------
        axis : Optional[Union[int, Tuple[int, ...]]]
            Axis or axes along which to operate. By default, flattened input is used.

        keepdims : bool, optional (default=False)
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        product_along_axis : mygrad.Tensor
            A tensor shaped as `a` but with the specified axis removed."""
        return Tensor._op(
            Prod, self, op_kwargs=dict(axis=axis, keepdims=keepdims), constant=constant
        )

    def cumprod(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Return the cumulative product of elements along a given axis.

        This docstring was adapted from the official numpy documentation

        Parameters
        ----------
        axis : Optional[int]
            Axis along which the cumulative product is computed.  By default
            the input is flattened.

        constant : bool, optional(default=False)
            If ``True``, the returned tensor is a constant (it
            does not back-propagate a gradient)

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        mygrad.Tensor

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is
        raised on overflow."""

        return Tensor._op(CumProd, self, op_kwargs=dict(axis=axis), constant=constant)

    def cumsum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Return the cumulative sum of the elements along a given axis.

        This docstring was adapted from the official numpy documentation

        Parameters
        ----------
        axis : int, optional
            Axis along which the cumulative sum is computed. The default
            (None) is to compute the cumsum over the flattened array.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        mygrad.Tensor
        """

        return Tensor._op(CumSum, self, op_kwargs=dict(axis=axis), constant=constant)

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Mean of tensor elements over a given axis.

        Parameters
        ----------
        x : ArrayLike

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

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        mean_along_axis : Tensor
            A Tensor with the same shape as `self`, with the specified
            axis/axes removed. If `self` is a 0-d tensor, or if `axis` is None,
            a 0-dim Tensor is returned.
        """
        return Tensor._op(
            Mean, self, op_kwargs=dict(axis=axis, keepdims=keepdims), constant=constant
        )

    def std(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Compute the standard deviation along the specified axis.

        Returns the variance of the array elements, a measure of the spread of a
        distribution.  The variance is computed for the flattened array by
        default, otherwise over the specified axis.

        Parameters
        ----------
        axis : Optional[Union[int, Tuple[int, ...]]]
            Axis or axes along which the variance is computed.  The default is to
            compute the variance of the flattened array.

        ddof : int, optional (default=0)
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By
            default `ddof` is zero.

        keepdims : bool, optional (default=False)
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        std : mygrad.Tensor

        Notes
        -----
        The variance is the average of the squared deviations from the mean,
        i.e.,  ``var = mean(abs(x - x.mean())**2)``.

        The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
        If, however, `ddof` is specified, the divisor ``N - ddof`` is used
        instead.  In standard statistical practice, ``ddof=1`` provides an
        unbiased estimator of the variance of a hypothetical infinite population.
        ``ddof=0`` provides a maximum likelihood estimate of the variance for
        normally distributed variables."""
        return Tensor._op(
            StdDev,
            self,
            op_kwargs=dict(axis=axis, keepdims=keepdims, ddof=ddof),
            constant=constant,
        )

    def var(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Compute the variance along the specified axis.

        Returns the variance of the array elements, a measure of the spread of a
        distribution.  The variance is computed for the flattened array by
        default, otherwise over the specified axis.

        Parameters
        ----------
        axis : Optional[int, Tuple[int, ...]]
            Axis or axes along which the variance is computed.  The default is to
            compute the variance of the flattened array.

        ddof : int, optional (default=0)
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By
            default `ddof` is zero.

        keepdims : bool, optional (default=False)
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array..

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.
        Returns
        -------
        variance : mygrad.Tensor

        Notes
        -----
        The variance is the average of the squared deviations from the mean,
        i.e.,  ``var = mean(abs(x - x.mean())**2)``.

        The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
        If, however, `ddof` is specified, the divisor ``N - ddof`` is used
        instead.  In standard statistical practice, ``ddof=1`` provides an
        unbiased estimator of the variance of a hypothetical infinite population.
        ``ddof=0`` provides a maximum likelihood estimate of the variance for
        normally distributed variables."""
        return Tensor._op(
            Variance,
            self,
            op_kwargs=dict(axis=axis, keepdims=keepdims, ddof=ddof),
            constant=constant,
        )

    def max(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Return the maximum of a tensor or maximum along its axes.

        Parameters
        ----------
        x : ArrayLike

        axis : Optional[int, Tuple[int, ...]]
            Axis or axes along which to operate. By default, flattened input is used.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `arr`.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        max : mygrad.Tensor
            Maximum of `a`. If `axis` is None, the result is a 0-D tensor.

        Examples
        --------
        >>> import mygrad as mg
        >>> import numpy as np
        >>> a = mg.arange(4).reshape((2,2))
        >>> a
        Tensor([[0, 1],
                [2, 3]])
        >>> mg.amax(a)           # Maximum of the flattened array
        Tensor(3)
        >>> mg.amax(a, axis=0)   # Maxima along the first axis
        Tensor([2, 3])
        >>> mg.amax(a, axis=1)   # Maxima along the second axis
        Tensor([1, 3])
        >>> b = mg.arange(5, dtype=float)
        >>> b[2] = np.NaN
        >>> mg.amax(b)
        Tensor(nan)
        """
        return Tensor._op(
            Max,
            self,
            op_kwargs=dict(axis=axis, keepdims=keepdims, dtype=_NoValue),
            constant=constant,
        )

    def min(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Return the minimum of a tensor or minimum along its axes.

        Parameters
        ----------
        axis : Optional[int, Tuple[int, ...]]
            Axis or axes along which to operate. By default, flattened input is used.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `arr`.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        min : mygrad.Tensor
            Minimum of `a`. If `axis` is None, the result is a 0-D tensor.

        Examples
        --------
        >>> import mygrad as mg
        >>> import numpy as np
        >>> a = mg.arange(4).reshape((2,2))
        >>> a
        Tensor([[0, 1],
                [2, 3]])
        >>> mg.amin(a)           # Minimum of the flattened array
        Tensor(0)
        >>> mg.amin(a, axis=0)   # Minima along the first axis
        Tensor([0, 1])
        >>> mg.amin(a, axis=1)   # Minima along the second axis
        Tensor([0, 2])
        >>> b = mg.arange(5, dtype=float)
        >>> b[2] = np.NaN
        >>> mg.amin(b)
        Tensor(nan)
        """
        return Tensor._op(
            Min,
            self,
            op_kwargs=dict(axis=axis, keepdims=keepdims, dtype=_NoValue),
            constant=constant,
        )

    def swapaxes(
        self, axis1: int, axis2: int, *, constant: Optional[bool] = None
    ) -> "Tensor":
        """Interchange two axes of a tensor.

        Parameters
        ----------
        axis1 : int
            First axis.

        axis2 : int
            Second axis.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        mygrad.Tensor
        """
        return Tensor._op(SwapAxes, self, op_args=(axis1, axis2), constant=constant)

    def transpose(
        self: ArrayLike, *axes: int, constant: Optional[bool] = None
    ) -> "Tensor":
        """Permute the dimensions of a tensor.

        Parameters
        ----------
        axes : int
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        mygrad.Tensor
            `a` with its axes permuted.  A new tensor is returned.

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.tensor([[1, 2], [3, 4]])
        >>> a
        Tensor([[1, 2],
                [3, 4]])
        >>> a.transpose()
        Tensor([[1, 3],
                [2, 4]])
        >>> a.transpose((1, 0))
        Tensor([[1, 3],
                [2, 4]])
        >>> a.transpose(1, 0)
        Tensor([[1, 3],
                [2, 4]])"""
        if not axes:
            axes = None
        elif hasattr(axes[0], "__iter__") or axes[0] is None:
            if len(axes) > 1:
                raise TypeError(
                    f"'{type(axes[0])}' object cannot be interpreted as an integer"
                )
            axes = axes[0]
        return Tensor._op(Transpose, self, op_args=(axes,), constant=constant)

    def moveaxis(
        self,
        source: Union[int, Tuple[int, ...]],
        destination: Union[int, Tuple[int, ...]],
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """Move axes of a tensor to new positions. Other axes remain in their
        original order.


        Parameters
        ----------
        source : Union[int, Sequence[int]]
            Original positions of the axes to move. These must be unique.

        destination : Union[int, Sequence[int]]
            Destination positions for each of the original axes. These must also be
            unique.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.

        Returns
        -------
        result : mygrad.Tensor
            Array with moved axes. This array is a view of the input array.."""
        return Tensor._op(
            MoveAxis, self, op_args=(source, destination), constant=constant
        )

    def squeeze(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        constant: Optional[bool] = None,
    ) -> "Tensor":
        """
        Remove single-dimensional entries from the shape of a tensor.

        This docstring was adapted from ``numpy.squeeze``

        Parameters
        ----------
        axis : Optional[int, Tuple[int, ...]]
            Selects a subset of the single-dimensional entries in the
            shape. If an axis is selected with shape entry greater than
            one, an error is raised.

        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.


        Returns
        -------
        mygrad.Tensor

        Raises
        ------
        ValueError
            If ``axis`` is not ``None``, and an axis being squeezed is not of length 1"""
        return Tensor._op(Squeeze, self, op_args=(axis,), constant=constant)

    def ravel(self, *, constant: Optional[bool] = None) -> "Tensor":
        """
        Flattens contents of a tensor into a contiguous 1-D array.  A copy is made only if needed.

        This docstring was adapted from ``numpy.ravel``.

        Parameters
        ----------
        constant : Optional[bool]
            If ``True``, this tensor is treated as a constant, and thus does not
            facilitate back propagation (i.e. ``constant.grad`` will always return
            ``None``).

            Defaults to ``False`` for float-type data.
            Defaults to ``True`` for integer-type data.

            Integer-type tensors must be constant.


        Returns
        -------
        mygrad.Tensor

        Notes
        -----
        ``ravel`` utilizes C-ordering, meaning that it reads & writes elements using
        C-like index ordering; the last axis index changing fastest, and, proceeding
        in reverse order, the first axis index changing slowest.
        """
        return Tensor._op(Ravel, self, constant=constant)

    def argmax(
        self, axis: Optional[int] = None, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns the indices of the maximum values along an axis.

        Parameters
        ----------
        a: array_like

        axis: int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.

        out: numpy.array, optional
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.

        Returns
        -------
        numpy.ndarray[int]"""

        return np.argmax(self.data, axis, out)

    def argmin(
        self, axis: Optional[int] = None, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns the indices of the minimum values along an axis.

        Parameters
        ----------
        axis: int, optional
            By default, the index is into the flattened array, otherwise along the specified axis.

        out: numpy.array, optional
            If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.

        Returns
        -------
        numpy.ndarray[int]"""

        return np.argmin(self.data, axis, out)

    def any(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Test whether any array or Tensor element along a given axis evaluates to True.

        Returns single boolean if `axis` is ``None``

        This documentation was adapted from ``numpy.add``

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a logical OR reduction is performed.
            The default (``axis=None``) is to perform a logical OR over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
            If this is a tuple of ints, a reduction is performed on multiple
            axes, instead of a single axis or all the axes as before.

        out : ndarray, optional
            Alternate output array in which to place the result.  It must have
            the same shape as the expected output and its type is preserved
            (e.g., if it is of type float, then it will remain so, returning
            1.0 for True and 0.0 for False, regardless of the type of `a`).
            See `ufuncs-output-type` for more details.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            If the default value is passed, then `keepdims` will not be
            passed through to the `any` method of sub-classes of
            `ndarray`, however any non-default value will be.  If the
            sub-class' method does not implement `keepdims` any
            exceptions will be raised.

        Returns
        -------
        any : bool or ndarray
            A new boolean or `ndarray` is returned unless `out` is specified,
            in which case a reference to `out` is returned.

        See Also
        --------
        Tensor.any : equivalent method

        """
        return np.any(self.data, axis=axis, out=out, keepdims=keepdims)
