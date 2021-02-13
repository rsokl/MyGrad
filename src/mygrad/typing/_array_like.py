import sys
from typing import TYPE_CHECKING, List, Sequence, Tuple, Union

import numpy as np

if sys.version_info >= (3, 8):
    from typing import Protocol

    HAS_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAS_PROTOCOL = False
        Protocol = object
    else:
        HAS_PROTOCOL = True

if TYPE_CHECKING:
    from mygrad import Tensor

if not TYPE_CHECKING and not HAS_PROTOCOL:

    class ImplementsArray:
        def __array__(self, dtype: None = ...) -> np.ndarray:
            ...


else:

    class ImplementsArray(Protocol):
        def __array__(self, dtype: None = ...) -> np.ndarray:
            ...


Real = Union[int, float]


sr1 = Sequence[Real]
sr2 = Sequence[sr1]
sr3 = Sequence[sr2]
sr4 = Sequence[sr3]
sr5 = Sequence[sr4]
sr6 = Sequence[sr5]

# Sequence[Union[s1, s2]] is *not* valid!
SequenceNDReals = Union[sr1, sr2, sr3, sr4, sr5, sr6]

# include Tensor and ndarray explicitly in case `ImplementsArray`
# is not protocol
ArrayLike = Union[Real, "Tensor", np.ndarray, ImplementsArray, SequenceNDReals]

sb1 = Sequence[Real]
sb2 = Sequence[sb1]
sb3 = Sequence[sb2]
sb4 = Sequence[sb3]
sb5 = Sequence[sb4]
sb6 = Sequence[sb5]


SequenceNDBools = Union[
    sb1, sb2, sb3, sb4, sb5, sb6
]  # Sequence[Union[s1, s2]] is *not* valid!


Mask = Union[ImplementsArray, np.ndarray, "Tensor", bool, SequenceNDBools]

si1 = Sequence[int]
si2 = Sequence[si1]
si3 = Sequence[si2]
si4 = Sequence[si3]
si5 = Sequence[si4]
si6 = Sequence[si5]


Index = Union[int, None, slice, ImplementsArray, Tuple["Index"], List["Index"]]
