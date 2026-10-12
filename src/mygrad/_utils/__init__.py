from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator, Iterable
from functools import wraps
from typing import TYPE_CHECKING, Generic, TypeVar
from weakref import ReferenceType

if TYPE_CHECKING:  # pragma: no cover
    from mygrad.operation_base import Operation  # noqa: F401
    from mygrad.tensor_base import Tensor

__all__ = [
    "collect_all_operations_and_clear_grads",
    "ContextTracker",
    "reduce_broadcast",
    "SkipGradient",
    "WeakRef",
    "WeakRefIterable",
]


T = TypeVar("T")


def collect_all_tensors_and_clear_grads(
    t: "Tensor",
    seen: set[int],
    topo_sorted_tensors: deque["Tensor"],
):
    """Accumulates in `seen` the ids of all non-constant tensors involved in
    creating `t` (including `t`), clearing the gradients of every tensor visited.

    `topo_sorted_tensors` is populated in topological order, with `t` first.

    `seen` and `topo_sorted_tensors` are updated in-place.

    This is a depth-first traversal that uses an explicit stack (rather than
    recursion) so that it can handle graphs that are deeper than Python's
    recursion limit."""
    # entries: (tensor, whether its inputs have already been traversed)
    stack = [(t, False)]
    while stack:
        t, inputs_done = stack.pop()
        if inputs_done:
            seen.add(id(t))
            topo_sorted_tensors.appendleft(t)
            continue

        t._view_grad = None
        t._grad = None

        if t._constant or id(t) in seen:
            continue

        stack.append((t, True))
        if t._creator is not None:
            # reversed, so that inputs are traversed in order
            stack.extend((var, False) for var in reversed(t._creator.variables))


if TYPE_CHECKING:  # pragma: no cover
    from weakref import ReferenceType as WeakRef
else:  # pragma: no cover

    class WeakRef(Generic[T]):
        __slots__ = ()

        def __init__(self, ob: T, callback=None, **annotations):  # pragma: no cover
            ...

        def __call__(self) -> None | T:  # pragma: no cover
            ...


class WeakRefIterable(Generic[T]):
    """
    Stores weakrefs in a list and, upon iteration, yields
    only living references.
    """

    __slots__ = ("data",)

    def __init__(self, data: Iterable[T] | None = None):
        self.data: list[WeakRef[T]] = []

        if data is not None:
            self.data: list[WeakRef[T]] = list(ReferenceType(x) for x in data)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __bool__(self):
        if len(self.data) == 0:
            return False
        else:
            return any(True for _ in self)

    def append(self, item: T):
        self.data.append(ReferenceType(item))

    def clear(self):
        self.data.clear()

    def __iter__(self) -> Generator[T, None, None]:
        for ref in self.data:
            item = ref.__call__()  # use __call__ to help type checker..
            if item is not None:
                yield item


class SkipGradient(Exception):
    """The gradient for the current tensor-label pair has already
    been computed, scaled, and back-propped, skip gradient calculation."""


def reduce_broadcast(grad, var_shape):
    """Sum-reduce axes of `grad` so its shape matches `var_shape.

    This the appropriate mechanism for backpropagating a gradient
    through an operation in which broadcasting occurred for the
    given variable.

    Parameters
    ----------
    grad : numpy.ndarray
    var_shape : Tuple[int, ...]

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        The dimensionality of the gradient cannot be less than
        that of its associated variable."""
    if grad.shape == var_shape:
        return grad

    if grad.ndim != len(var_shape):
        if grad.ndim < len(var_shape):
            raise ValueError(
                f"The dimensionality of the gradient of the broadcasted "
                f"operation ({grad.ndim}) is less than that of its associated variable "
                f"({len(var_shape)})"
            )
        grad = grad.sum(axis=tuple(range(grad.ndim - len(var_shape))))

    keepdims = tuple(n for n, i in enumerate(grad.shape) if i != var_shape[n])
    if keepdims:
        grad = grad.sum(axis=keepdims, keepdims=True)

    return grad


class ContextTracker(ABC):
    """A context manager and decorator for managing a boolean
    global state"""

    # tracks context depth
    _depth = 0  # type: int

    # the value that the state is set to upon entering the context
    _enter_set_value: bool | None = None

    @property
    @abstractmethod
    def state(self) -> bool:  # pragma: no cover
        raise NotImplementedError()

    @state.setter
    @abstractmethod
    def state(self, value: bool):  # pragma: no cover
        raise NotImplementedError()

    def __init__(self):
        # keeps track of what MemGuard was at a given depth
        self._depth_tracker: dict[int, bool] = dict()

    def __bool__(self) -> bool:  # pragma: no cover
        return self.state

    def __enter__(self):
        self._depth_tracker[self._depth] = self.state
        self._depth += 1
        self.state = self._enter_set_value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._depth -= 1
        self.state = self._depth_tracker.pop(self._depth)

    def __call__(self, func: Callable) -> Callable:
        """Decorates a function within the context"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
