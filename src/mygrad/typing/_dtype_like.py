from typing import Type, Union

from numpy import dtype

DTypeLike = Union[
    dtype,
    None,
    type,
    str,
]

DTypeLikeReals = Union[
    dtype,
    None,
    Type[int],
    Type[float],
    str,
]
