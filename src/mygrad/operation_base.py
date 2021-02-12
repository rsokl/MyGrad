"""
Defines the base class for mathematical operations capable of back-propagating
gradients to their input tensors."""
from abc import ABC, abstractmethod
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple, Union
from weakref import ReferenceType

import numpy as np

from mygrad._utils import SkipGradient, reduce_broadcast
from mygrad.errors import InvalidBackprop, InvalidGradient

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor
    from mygrad._utils import WeakRef

__all__ = [
    "Operation",
    "Ufunc",
    "UnaryUfunc",
    "BinaryUfunc",
    "Sequential",
]


class _NoValueType:
    """Special keyword value.

    The instance of this class may be used as the default value assigned to a
    deprecated keyword in order to check if it has been given a user defined
    value.
    """

    __instance = None

    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super(_NoValueType, cls).__new__(cls)
        return cls.__instance

    def __repr__(self):  # pragma: no cover
        return "<no value>"


_NoValue = _NoValueType()


class Operation(ABC):
    """Base class for all tensor operations that support back-propagation
    of gradients.

    Consider the Operation-instance ``f``. A forward-pass through ``f`` is defined
    via ``f.__call__(...)``. Thus, given tensors ``a`` and ``b``, a computational
    graph is defined ``f.__call__(a, b) -> c``, where the "creator" of tensor ``c``
    is recorded as ``f``::

          (node: a) --+
                       -> [operation: f(a, b)] --> (node: c)
          (node: b) --+

    Back-propagating through ``c`` will instruct ``f`` to back-propagate
    the gradient to its inputs, which are recorded as ``a`` and ``b``. Each
    node then back-propagates to any Operation-instance that is recorded
    as its creator, and so on.
    """

    # Can be set to true if the operation is guaranteed to not returns a view
    # this will reduce some overhead on checking for shared memory
    can_return_view = False  # type: bool

    # Stores the input tensors that the operation will backprop through.
    variables: Tuple["Tensor", ...]

    def __init__(self):
        # Stores positional and keyword arguments used to call op.
        # Can be set optionally - only if op needs to be "replayed",
        # e.g. with a view
        self.replay_args: Optional[Tuple[Any, ...]] = None
        self.replay_kwargs: Optional[Dict[str, Any]] = None
        self.replay_force_constant: Optional[bool] = None
        self.where: Union[bool, np.ndarray] = True

    @staticmethod
    def grad_post_process_fn(
        grad: np.ndarray, var_shape: Tuple[int, ...]
    ) -> np.ndarray:
        # this function gets called all of the time; we can avoid
        # the extra function call by doing the shape check upfront
        if grad.shape == var_shape:
            return grad
        return reduce_broadcast(grad, var_shape)

    @abstractmethod
    def __call__(self, *input_vars: "Tensor", **kwargs) -> np.ndarray:
        """Performs a forward pass, f, of this Operation::

                         f(x1, ...., xn)

        Parameters
        ----------
        *input_vars : mygrad.Tensor
            The input-arguments of f. The tuple (x1, ...., xn)
            should be bound to the instance-attribute `self.variables`

        **kwargs : Any
            Additional arguments for the operation

        Returns
        -------
        numpy.ndarray
            The output of the forward pass function.

        Notes
        -----
        This method should set the ``self.variables`` attribute
        with a tuple storing all of the input tensors of this operations"""

        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def backward_var(self, grad: np.ndarray, index: int, **kwargs) -> np.ndarray:
        """Given ``grad = dℒ/df``, computes ``∂ℒ/∂x_{i}``, where ``x_{i}`` is one
        of ``x1, ...., xn``.

        ``ℒ`` is assumed to be the terminal node from which ``ℒ.backward()`` was
        called.

        Parameters
        ----------
        grad : numpy.ndarray
            The back-propagated total derivative with respect to the present
            operation: dℒ/df. This will have the same shape as the result
            of the forward pass.

        index : int
            The index-location of ``var`` in ``self.variables``

        Returns
        -------
        numpy.ndarray
            ∂ℒ/∂x_{i}

        Raises
        ------
        SkipGradient"""
        raise NotImplementedError()  # pragma: no cover

    def backward(
        self,
        grad: np.ndarray,
        *,
        graph: Set["WeakRef[Operation]"],
        **kwargs,
    ):
        """Back-propagates the gradient through all of the operation's inputs,
        which are stored in the tuple `self.variables`.

        Constant tensors (`tensor.constant is True`) skipped by this process.

        Parameters
        ----------
        grad : numpy.ndarray
            The back-propagated total derivative with respect to the present
            operation (`f`): d(out)/df

        graph : Set[Operation]
            The set of all operations relevant to the terminal node of the computational graph,
            which triggered back-propagation.
        """
        for index, var in enumerate(self.variables):
            if not var.constant:
                if self.where is not True:  # pragma: no cover
                    raise NotImplementedError(
                        "Backprop is not supported for operations that "
                        "were filtered through a `where` boolean mask."
                    )
                if not var._ops:
                    raise InvalidBackprop(
                        f"Part of the computational graph containing "
                        f"this tensor, {var}, was 'cleared' prior to backprop.\n"
                        f"It is recommended that you clear all computational graphs "
                        f"and restart your computation."
                    )

                try:
                    # don't cast to array here so that we have an easier time
                    # doing type checking (e.g. avoid `None` -> `array(None, dtype=obj)`
                    backed_grad = self.backward_var(grad, index, **kwargs)
                except SkipGradient:
                    continue

                if not isinstance(backed_grad, (np.ndarray, np.number, Real)):
                    raise InvalidGradient(
                        f"An invalid gradient-value was passed to:"
                        f"\n\t`{type(self).__name__}.backward_var(<gradient>, index={index})`"
                        f"\nGradients are expected to be real-valued scalars or "
                        f"numpy arrays, got a gradient of type: {type(backed_grad)}"
                    )

                backed_grad = np.array(backed_grad, copy=False)
                backed_grad = self.grad_post_process_fn(backed_grad, var.shape)

                if var._grad is None:
                    backed_grad = (
                        np.copy(backed_grad)
                        # `backed_grad` is view of grad; we want to be able to
                        # augment tmp-grad inplace later
                        if backed_grad.base is not None or (backed_grad is grad)
                        else backed_grad
                    )
                    if backed_grad.dtype != var.dtype:
                        backed_grad = backed_grad.astype(var.dtype, copy=False)

                    var._grad = backed_grad
                else:
                    var._grad += backed_grad

        # Avoid visiting the same node multiple times. Note that we don't store
        # these by the node itself, since Tensors are unhashable, but by its `id`.
        visited = set()
        ref_op = ReferenceType(self)

        for var in (
            i for i in self.variables if not i.constant and i.creator is not None
        ):
            var_id = id(var)
            if var_id in visited:
                continue
            visited.add(var_id)
            var._accum_ops.add(ref_op)
            var._backward(graph=graph)


class Ufunc(Operation, ABC):
    """The base class for mygrad's universal functions.

    'A universal function (or ufunc for short) is a function that operates on
    ndarrays in an element-by-element fashion, supporting array broadcasting, type casting,
    and several other standard features. That is, a ufunc is a “vectorized” wrapper for a
    function that takes a fixed number of specific inputs and produces a fixed number of
    specific outputs.' [1]_

    References
    ----------
    .. [1] Retrieved from https://numpy.org/doc/stable/reference/ufuncs.html"""

    @property
    @abstractmethod
    def numpy_ufunc(self) -> np.ufunc:
        raise NotImplementedError()  # pragma: no cover

    @property
    def nin(self) -> int:
        """The number of inputs."""
        return self.numpy_ufunc.nin

    @property
    def nout(self) -> int:
        """The number of outputs."""
        return self.numpy_ufunc.nout

    @property
    def nargs(self) -> int:
        """The number of arguments."""
        return self.numpy_ufunc.nargs

    @property
    def ntypes(self) -> int:
        """The number of types."""
        return self.numpy_ufunc.ntypes

    @property
    def types(self):
        """Returns a list with types grouped input->output."""
        return self.numpy_ufunc.types

    @property
    def identity(self):
        """The identity value."""
        return self.numpy_ufunc.identity

    @property
    def signature(self):
        """Definition of the core elements a generalized ufunc operates on."""
        return self.numpy_ufunc.signature


class UnaryUfunc(Ufunc, ABC):
    """A base class that specifies the common interface to – and facilitates
    back-prop through – ufuncs that operate on a single array argument;
    e.g. `mygrad.sin`, `mygrad.negative`."""

    def __call__(
        self,
        x1: "Tensor",
        out: Optional[np.ndarray] = None,
        *,
        where: Union[bool, np.ndarray] = True,
        dtype=None,
    ) -> np.ndarray:
        """f(x1, out=None, *, where=True, dtype=None)

        Parameters
        ----------
        x1 : Tensor, shape-(...)
            The input to the operation.

            This tensor is saved to the state of the operation instance
            so that back-prop can be performed through it.

        out : Optional[np.ndarray]
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        where: Union[bool, np.ndarray]
            Accepts a boolean array which is broadcast together with ``x1``.
            Values of True indicate to calculate the ufunc at that position, values
            of False indicate to leave the value in the output alone.

        dtype : Optional[numpy.dtype, str, object]
            Overrides the dtype of the calculation and output array.

        Returns
        -------
        y : ndarray, shape-(...)
            A numpy array of the same shape as ``x1`` with the ufunc applied
            elementwise on ``x1``.

        Notes
        -----
        This docstring was adapted from numpy's documentation [1]_.

        References
        ----------
        .. [1] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        self.variables: Tuple["Tensor"] = (x1,)
        if where is not True:
            self.where = where
        return self.numpy_ufunc(x1.data, out=out, where=where, dtype=dtype)


class BinaryUfunc(Ufunc, ABC):
    """A base class that specifies the common interface to – and facilitates
    back-prop through – mygrad's ufuncs that operate on a two array arguments;
    e.g. `mygrad.add`, `mygrad.multiply`.
    """

    def __call__(
        self,
        x1: "Tensor",
        x2: "Tensor",
        out: Optional[np.ndarray] = None,
        *,
        where: Union[bool, np.ndarray, _NoValueType] = True,
        dtype=None,
    ) -> np.ndarray:
        """f(x1, x2, out=None, *, where=True, dtype=None)

        Parameters
        ----------
        x1 : Tensor
            The first input to the operation.

            This tensor is saved to the state of the operation instance
            so that back-prop can be performed through it.

        x2 : Tensor
            The second input to the operation.

            This tensor is saved to the state of the operation instance
            so that back-prop can be performed through it.

        out : Optional[np.ndarray]
            A location into which the result is stored. If provided, it must
            have a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned.

        where: Union[bool, np.ndarray]
            Accepts a boolean array which is broadcast jointly with ``x1`` and ``x2``.
            Values of True indicate to calculate the ufunc at that position, values
            of False indicate to leave the value in the output alone.

        dtype : Optional[numpy.dtype, str, object]
            Overrides the dtype of the calculation and output array.

        Returns
        -------
        y : ndarray
            A numpy array resulting from the elementwise application of the ufunc to
            corresponding pairs of elements from ``x1`` and ``x2``, respectively.

            If ``x1`` and ``x2`` are of different shapes, then the operation is broadcast
            across them [1]_.

        Notes
        -----
        This docstring was adapted from numpy's documentation [2]_.

        References
        ----------
        .. [1] https://numpy.org/doc/stable/user/basics.broadcasting.html
        .. [2] Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.add.html
        """
        self.variables: Tuple["Tensor", "Tensor"] = (x1, x2)
        if where is not True and where is not _NoValue:
            self.where = where
            return self.numpy_ufunc(x1.data, x2.data, out=out, where=where, dtype=dtype)
        else:
            return self.numpy_ufunc(x1.data, x2.data, out=out, dtype=dtype)


class Sequential(Operation, ABC):
    """A base class that specifies the common interface to – and facilitates
    back-prop through – numpy's sequential functions; e.g. `numpy.sum`, `numpy.var`,
    `numpy.max`"""

    numpy_func: Callable[..., np.ndarray]
    _integer_axis_only: bool = False

    @staticmethod
    @abstractmethod
    def numpy_func(
        a: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype=None,
        out: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError()  # pragma: no cover

    def __init__(self):
        self.axis: Optional[Union[int, Tuple[int, ...]]]
        self.keepdims: Optional[bool]
        self.initial: Real
        self.out_shape: Tuple[int, ...]
        super().__init__()

    def __call__(
        self,
        a: "Tensor",
        axis=None,
        dtype=None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = _NoValue,
        initial: Real = _NoValue,
        *,
        where: Union[bool, np.ndarray] = _NoValue,
        ddof: int = _NoValue,
    ) -> np.ndarray:
        self.variables: Tuple["Tensor"] = (a,)

        if where is not True:
            self.where = where

        self.keepdims = keepdims
        self.initial = initial
        self.ddof = ddof

        # Unless axis is None or the op is integer-axis-only
        # normalize axis to be a tuple of ints.
        if (
            not self._integer_axis_only
            and axis is not None
            and not hasattr(axis, "__iter__")
        ):
            self.axis = (axis,)
        else:
            self.axis = axis

        kwargs = {}

        if keepdims is not _NoValue:
            kwargs["keepdims"] = keepdims

        if initial is not _NoValue:  # pragma: no cover
            kwargs["initial"] = initial

        if where is not _NoValue:
            kwargs["where"] = where

        if ddof is not _NoValue:
            kwargs["ddof"] = ddof

        if dtype is not _NoValue:
            kwargs["dtype"] = dtype

        out = self.numpy_func(a.data, axis=axis, out=out, **kwargs)
        self.out_shape = out.shape

        return out
