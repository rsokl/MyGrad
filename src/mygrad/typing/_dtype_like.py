from typing import TYPE_CHECKING, Type, TypeVar, Union

from numpy import dtype

if TYPE_CHECKING:  # pragma: no cover
    DTypeLike = Union[dtype, None, type, str]
else:  # pragma: no cover
    DTypeLike = TypeVar(
        "DTypeLike",
        bound=Union[dtype, None, type, str],
    )

if TYPE_CHECKING:  # pragma: no cover
    DTypeLikeReals = Union[dtype, None, Type[bool], Type[int], Type[float]]

else:  # pragma: no cover

    class DTypeLikeReals: ...
