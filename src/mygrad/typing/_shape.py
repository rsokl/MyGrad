from typing import TYPE_CHECKING, Tuple, TypeVar

if TYPE_CHECKING:
    Shape = Tuple[int, ...]
else:
    Shape = TypeVar("Shape", bound=Tuple[int, ...])
