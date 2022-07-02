from abc import ABC, abstractmethod
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)
from weakref import ReferenceType

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor
    from mygrad.operation_base import Operation

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
    seen: Set[int],
    topo_sorted_tensors: Deque["Tensor"],
    _marked: Optional[Set[int]] = None,
):
    """Recursively accumulates in `seen` all operations involved
    in creating `t`.

    `seen` is updated in-place
    """
    t._view_grad = None
    t._grad = None

    if _marked is None:
        _marked = set()

    if t.constant:
        return

    id_ = id(t)

    if id_ in seen:
        return

    if id_ in _marked:  # pragma: no cover
        assert False, "Computational graph is contains a cycle"

    _marked.add(id_)

    if t.creator is not None:
        for t_loop in t.creator.variables:
            collect_all_tensors_and_clear_grads(t_loop, seen, topo_sorted_tensors)
        del t_loop

    _marked.remove(id_)
    seen.add(id_)
    topo_sorted_tensors.appendleft(t)


if TYPE_CHECKING:  # pragma: no cover
    from weakref import ReferenceType as WeakRef
else:  # pragma: no cover

    class WeakRef(Generic[T]):
        __slots__ = ()

        def __init__(self, ob: T, callback=None, **annotations):  # pragma: no cover
            ...

        def __call__(self) -> Union[None, T]:  # pragma: no cover
            ...


class WeakRefIterable(Generic[T]):
    """
    Stores weakrefs in a list and, upon iteration, yields
    only living references.
    """

    __slots__ = ("data",)

    def __init__(self, data: Optional[Iterable[T]] = None):
        self.data: List[WeakRef[T]] = []

        if data is not None:
            self.data: List[WeakRef[T]] = list(ReferenceType(x) for x in data)

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
    _enter_set_value: Optional[bool] = None

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
        self._depth_tracker: Dict[int, bool] = dict()

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
