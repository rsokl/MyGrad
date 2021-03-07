from typing import TypeVar

import numpy as np

from mygrad.tensor_base import Tensor
from mygrad.typing import ArrayLike, Mask

T = TypeVar("T", np.ndarray, Tensor)


def copyto(dst: T, src: ArrayLike, casting="same_kind", where: Mask = True) -> T:
    if isinstance(dst, Tensor):
        dst = dst.data

    if isinstance(src, Tensor):
        src = src.data
