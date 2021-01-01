"""
Defines the base class for mathematical operations capable of back-propagating
gradients to their input tensors."""
from abc import ABC, abstractmethod
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple
from weakref import ReferenceType

import numpy as np

from mygrad._utils import SkipGradient, reduce_broadcast
from mygrad.errors import InvalidBackprop, InvalidGradient

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor
    from mygrad._utils import WeakRef

__all__ = ["Operation", "BroadcastableOp"]


class Operation(ABC):
    """ Base class for all tensor operations that support back-propagation
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

    If an operation class has `scalar_only=True`, then the terminal node of a
    computational graph involving that operation can only trigger back-propagation
    from a 0-dimensional tensor (i.e. a scalar). This is `False` for operations that
    manifest as trivial element-wise operations over tensors. In such cases, the
    gradient of the operation can also be treated element-wise, and thus be computed
    unambiguously.
    """

    # tracks if a given operation-instance performs a
    # non-vectorized or broadcasted operation , which
    # requires that backpropagation be invoked from a scalar
    scalar_only = False  # type: bool

    # can be set to true if the operation is guaranteed to not returns a view
    # this will reduce some overhead on checking for shared memory
    can_return_view = False  # type: bool

    def __init__(self):
        # Stores positional and keyword arguments used to call op.
        # Can be set optionally - only if op needs to be "replayed",
        # e.g. with a view
        self.replay_args: Optional[Tuple[Any, ...]] = None
        self.replay_kwargs: Optional[Dict[str, Any]] = None
        self.replay_force_constant: Optional[bool] = None

    @abstractmethod
    def __call__(self, *input_vars: "Tensor", **kwargs) -> np.ndarray:
        """ Performs a forward pass, f, of this Operation::

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
        """ Given ``grad = dℒ/df``, computes ``∂ℒ/∂x_{i}``, where ``x_{i}`` is one
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
        _reduction: Optional[
            Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
        ] = None,
        **kwargs,
    ):
        """ Back-propagates the gradient through all of the operation's inputs.
        Constant tensors do not propagate a gradient.

        Parameters
        ----------
        grad : numpy.ndarray
            The back-propagated total derivative with respect to the present
            operation (`f`): d(out)/df

        graph : Set[Operation]
            The set of all operations relevant to the terminal node of the computational graph,
            which triggered back-propagation.

        _reduction : Optional[Callable[[ndarray, Tuple[int, ...]], ndarray]]
            Developer option-only. A callable used to process the gradient
            prior to accumulation (e.g. broadcast-reduction)
        """
        for index, var in enumerate(self.variables):
            if not var.constant:
                if not var._ops:
                    raise InvalidBackprop(
                        f"Part of the computational graph containing "
                        f"this tensor, {var}, was 'cleared' prior to backprop.\n"
                        f"It is recommended that you clear all computational graphs "
                        f"and restart your computation."
                    )

                try:
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
                if var._grad is None:
                    tmp_grad = np.asarray(backed_grad)

                    if _reduction is not None:
                        tmp_grad = _reduction(tmp_grad, var.shape)

                    var._grad = (
                        np.copy(tmp_grad)
                        # tmp-grad is view of grad; we want to be able to
                        # augment tmp-grad inplace later
                        if tmp_grad.base is not None or (tmp_grad is grad)
                        else tmp_grad
                    )
                else:

                    if _reduction is not None:
                        backed_grad = _reduction(backed_grad, var.shape)
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


class BroadcastableOp(Operation, ABC):
    """ Signals that an Operation's forward pass can broadcast its tensor arguments."""

    def backward(
        self,
        grad: np.ndarray,
        *,
        graph: Set["WeakRef[Operation]"],
        _reduction: Optional[
            Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
        ] = None,
        **kwargs,
    ):
        return super().backward(grad, graph=graph, _reduction=reduce_broadcast)
