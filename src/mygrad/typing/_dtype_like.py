from typing import Type, TypeVar, Union

from numpy import dtype

DTypeLike = TypeVar(
    "DTypeLike",
    bound=Union[
        dtype,
        None,
        type,
        str,
    ],
)

DTypeLikeReals = TypeVar(
    "DTypeLikeReals",
    bound=Union[
        dtype,
        None,
        Type[bool],
        Type[int],
        Type[float],
        str,
    ],
)
