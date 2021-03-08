import sys
from typing import TYPE_CHECKING, Optional, Tuple, Union

from numpy import dtype, ndarray

from mygrad.tensor_base import _REGISTERED_NO_DIFF_NUMPY_FUNCS
from mygrad.typing import ArrayLike

__all__ = [
    "allclose",
    "bincount",
    "can_cast",
    "copyto",
    "may_share_memory",
    "min_scalar_type",
    "result_type",
    "shares_memory",
    "shape",
]

_module = sys.modules[__name__]

if TYPE_CHECKING:  # pragma: no cover

    def allclose(
        a: ArrayLike,
        b: ArrayLike,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        pass

    def bincount(
        x: ArrayLike, weights: Optional[ArrayLike] = None, minlength: int = 0
    ) -> ndarray:
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

    def min_scalar_type(a: ArrayLike) -> dtype:
        pass

    def result_type(*arrays_and_dtypes: Union[ArrayLike, dtype]) -> dtype:
        pass

    def shares_memory(
        a: ArrayLike, b: ArrayLike, max_work: Optional[int] = None
    ) -> bool:
        pass

    def shape(a: ArrayLike) -> Tuple[int, ...]:
        pass


for _func in _REGISTERED_NO_DIFF_NUMPY_FUNCS:
    setattr(_module, _func.__name__, _func)
