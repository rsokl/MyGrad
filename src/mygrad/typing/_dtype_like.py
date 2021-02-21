from typing import TYPE_CHECKING, Type, TypeVar, Union

from numpy import dtype

if TYPE_CHECKING:
    DTypeLike = Union[dtype, None, type, str]
else:
    DTypeLike = TypeVar(
        "DTypeLike",
        bound=Union[dtype, None, type, str],
    )

if TYPE_CHECKING:
    DTypeLikeReals = Union[dtype, None, Type[bool], Type[int], Type[float], str]

else:
    DTypeLikeReals = TypeVar(
        "DTypeLikeReals",
        bound=Union[dtype, None, Type[bool], Type[int], Type[float], str],
    )
