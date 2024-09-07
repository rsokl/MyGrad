from typing import TYPE_CHECKING, Tuple, TypeAlias

from typing_extensions import TypeAlias

if TYPE_CHECKING:  # pragma: no cover
    Shape: TypeAlias = Tuple[int, ...]
else:

    class Shape: ...
