from typing import TYPE_CHECKING, Tuple

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    Shape: TypeAlias = Tuple[int, ...]
else:

    class Shape:
        ...
