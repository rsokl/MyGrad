from typing import TYPE_CHECKING, Tuple, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    Shape = Tuple[int, ...]
else:  # pragma: no cover
    Shape = TypeVar("Shape", bound=Tuple[int, ...])
