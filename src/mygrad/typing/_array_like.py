import sys
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor

from typing import Protocol

if sys.version_info >= (3, 10):  # pragma: no cover
    from types import EllipsisType
elif TYPE_CHECKING:  # pragma: no cover
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)


@runtime_checkable
class ImplementsArray(Protocol):
    def __array__(
        self, dtype: None = ..., copy: Optional[bool] = ...
    ) -> np.ndarray: ...


Real = Union[int, float]


sr1 = Sequence[Real]
sr2 = Sequence[sr1]
sr3 = Sequence[sr2]
sr4 = Sequence[sr3]

# Sequence[Union[s1, s2]] is *not* valid!
SequenceNDReals = Union[sr1, sr2, sr3, sr4]

# include Tensor and ndarray explicitly in case `ImplementsArray`
# is not protocol

if TYPE_CHECKING:  # pragma: no cover
    ArrayLike = Union[Real, "Tensor", np.ndarray, ImplementsArray, SequenceNDReals]
else:  # pragma: no cover

    class ArrayLike: ...


sb1 = Sequence[bool]
sb2 = Sequence[sb1]
sb3 = Sequence[sb2]
sb4 = Sequence[sb3]

# Sequence[Union[s1, s2]] is *not* valid!
SequenceNDBools = Union[sb1, sb2, sb3, sb4]

if TYPE_CHECKING:  # pragma: no cover
    Mask = Union[ImplementsArray, np.ndarray, "Tensor", bool, SequenceNDBools]
else:  # pragma: no cover
    Mask = TypeVar(
        "Mask",
        bound=Union[ImplementsArray, np.ndarray, "Tensor", bool, SequenceNDBools],
    )


Index = Union[
    int,
    None,
    slice,
    ImplementsArray,
    np.ndarray,
    EllipsisType,
    Sequence[int],
    Tuple["Index", ...],
    List["Index"],
]
