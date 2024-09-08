import itertools
import sys
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy

from mygrad.tensor_base import (
    _REGISTERED_BOOL_ONLY_UFUNC,
    _REGISTERED_CONST_ONLY_UFUNC,
    _REGISTERED_NO_DIFF_NUMPY_FUNCS,
)
from mygrad.typing import ArrayLike, DTypeLike, Mask

__all__ = [
    "allclose",
    "bincount",
    "can_cast",
    "ceil",
    "copyto",
    "divmod",
    "equal",
    "floor",
    "floor_divide",
    "fmod",
    "greater",
    "greater_equal",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "may_share_memory",
    "min_scalar_type",
    "mod",
    "not_equal",
    "remainder",
    "result_type",
    "rint",
    "shape",
    "shares_memory",
    "sign",
    "signbit",
    "trunc",
]

_module = sys.modules[__name__]


for _func in itertools.chain.from_iterable(
    (
        _REGISTERED_NO_DIFF_NUMPY_FUNCS,
        _REGISTERED_BOOL_ONLY_UFUNC,
        _REGISTERED_CONST_ONLY_UFUNC,
    )
):
    setattr(_module, _func.__name__, _func)

mod = numpy.remainder

if TYPE_CHECKING:  # pragma: no cover

    def allclose(
        a: ArrayLike,
        b: ArrayLike,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        pass

    def isclose(
        a: ArrayLike,
        b: ArrayLike,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        pass

    def bincount(
        x: ArrayLike, weights: Optional[ArrayLike] = None, minlength: int = 0
    ) -> numpy.ndarray:
        pass

    def can_cast(from_, to, casting="safe") -> bool:
        pass

    def copyto(
        dst: ArrayLike, src: ArrayLike, casting: str = "same_kind", where: bool = True
    ):
        pass

    def may_share_memory(
        a: ArrayLike, b: ArrayLike, max_work: Optional[int] = None
    ) -> bool:
        pass

    def min_scalar_type(a: ArrayLike) -> numpy.dtype:
        pass

    def result_type(*arrays_and_dtypes: Union[ArrayLike, numpy.dtype]) -> numpy.dtype:
        pass

    def shares_memory(
        a: ArrayLike, b: ArrayLike, max_work: Optional[int] = None
    ) -> bool:
        pass

    def shape(a: ArrayLike) -> Tuple[int, ...]:
        pass

    def isnan(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def isfinite(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def isinf(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def isnat(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def signbit(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def logical_not(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def logical_and(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def logical_or(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def logical_xor(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def greater(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def greater_equal(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def less(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def less_equal(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def equal(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def not_equal(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Union[numpy.ndarray, bool]: ...

    def floor_divide(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...

    def remainder(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...

    def divmod(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]: ...

    def mod(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]: ...

    def fmod(
        x1: ArrayLike,
        x2: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]: ...

    def rint(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...

    def sign(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...

    def floor(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...

    def ceil(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...

    def trunc(
        x: ArrayLike,
        out: Optional[ArrayLike] = None,
        *,
        casting: str = "same_kind",
        where: Mask = True,
        order: str = "K",
        dtype: DTypeLike = None,
        subok: bool = True,
    ) -> numpy.ndarray: ...
