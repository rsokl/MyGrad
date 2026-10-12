from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:  # pragma: no cover
    Shape: TypeAlias = tuple[int, ...]
else:

    class Shape: ...
