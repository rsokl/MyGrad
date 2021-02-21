import sys
from typing import TYPE_CHECKING, List, Sequence, Tuple, TypeVar, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor

if sys.version_info >= (3, 8):  # pragma: no cover
    from typing import Protocol

    HAS_PROTOCOL = True
else:  # pragma: no cover
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAS_PROTOCOL = False
        Protocol = object
    else:
        HAS_PROTOCOL = True


if not TYPE_CHECKING and not HAS_PROTOCOL:  # pragma: no cover

    class ImplementsArray:
        def __array__(self, dtype: None = ...) -> np.ndarray:
            ...


else:  # pragma: no cover

    class ImplementsArray(Protocol):
        def __array__(self, dtype: None = ...) -> np.ndarray:
            ...


Real = Union[int, float]


sr1 = Sequence[Real]
sr2 = Sequence[sr1]
sr3 = Sequence[sr2]
sr4 = Sequence[sr3]

# Sequence[Union[s1, s2]] is *not* valid!
SequenceNDReals = Union[sr1, sr2, sr3, sr4]

# include Tensor and ndarray explicitly in case `ImplementsArray`
# is not protocol

if TYPE_CHECKING:
    ArrayLike = Union[Real, "Tensor", np.ndarray, ImplementsArray, SequenceNDReals]
else:
    ArrayLike = TypeVar(
        "ArrayLike", Real, "Tensor", np.ndarray, ImplementsArray, SequenceNDReals
    )


sb1 = Sequence[bool]
sb2 = Sequence[sb1]
sb3 = Sequence[sb2]
sb4 = Sequence[sb3]

# Sequence[Union[s1, s2]] is *not* valid!
SequenceNDBools = Union[sb1, sb2, sb3, sb4]

if TYPE_CHECKING:
    Mask = Union[ImplementsArray, np.ndarray, "Tensor", bool, SequenceNDBools]
else:
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
    Sequence[int],
    Tuple["Index"],
    List["Index"],
]
